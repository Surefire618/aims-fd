import io
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyfhiaims
import pytest
from pyfhiaims.outputs.stdout import AimsStdout

import aims_fd
from aims_fd.cli import main
from aims_fd.control import comment_out, requests
from aims_fd.deformations import plan, run_ranges
from aims_fd.energies import (
    EnergyComponent,
    collect,
    decomposition,
    pyfhiaims_value,
    with_rest,
)
from aims_fd.finite_differences import fd_forces

from .fake_aims import E0, FORCES, FORCES_VDW, SIGMA, SIGMA_VDW

FAKE = f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).parent / 'fake_aims.py'))}"
SPECIES = "\n  species        Si\n    nucleus             14\n    mass                28.0855\n"
CONTROL = ("xc                 pbe\noccupation_type    gaussian 0.01\nk_grid             10 10 10\n"
           "sc_accuracy_rho    1e-7\nsc_accuracy_etot   1e-9\ncompute_forces     .true.\n"
           "compute_analytical_stress .true.\n" + SPECIES)
GEOMETRY = ("lattice_vector 0.0 2.7155 2.7155\nlattice_vector 2.7155 0.0 2.7155\n"
            "lattice_vector 2.7155 2.7155 0.0\natom 0.0 0.0 0.0 Si\n"
            "atom 1.424186383883 1.397611830330 1.377680915165 Si\n")


@pytest.fixture
def calc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FAKE_AIMS_LOG", str(tmp_path / "calls.log"))
    monkeypatch.delenv("FAKE_AIMS_FAIL", raising=False)
    (tmp_path / "control.in").write_text(CONTROL)
    (tmp_path / "geometry.in").write_text(GEOMETRY)
    (tmp_path / ".aimsfdrc").write_text(f"[machine]\naims_command = {FAKE}\n")
    return tmp_path


def parse(calc, text):
    (calc / "aims.out").write_text(text)
    return AimsStdout(calc / "aims.out")


def calls(calc):
    log = calc / "calls.log"
    return log.read_text().split() if log.exists() else []


def test_requests_and_comment_out():
    assert requests(CONTROL) == (True, True, 1e-4)
    assert requests("compute_forces .false.\ncompute_numerical_stress T\n"
                    "delta_numerical_stress 2d-4\n") == (False, True, 2e-4)
    assert requests("# compute_forces .true.\n") == (False, False, 1e-4)
    edited = comment_out(CONTROL, ("compute_forces", "compute_analytical_stress"))
    assert "# aims-fd removed: compute_forces     .true." in edited
    assert edited.count("\n") == CONTROL.count("\n")
    assert all(line.startswith("#") or "compute" not in line for line in edited.splitlines())


def test_aims_fd_writes_an_aims_out_that_pyfhiaims_reads(calc, capsys):
    assert main([]) == 0
    # reference + 2 atoms x 3 axes x 2 signs + 6 components x 2 signs
    assert len(calls(calc)) == 25
    text = capsys.readouterr().out
    out = parse(calc, text)
    assert f"  aims-fd version       : {aims_fd.__version__}\n" in text
    assert f"  pyfhiaims version     : {pyfhiaims.__version__}\n" in text
    assert out.aims_version == "260824"
    assert out.is_finished_ok
    assert out.converged
    np.testing.assert_allclose(out.forces, FORCES, atol=1e-6)
    np.testing.assert_allclose(out.stress, SIGMA, atol=1e-9)
    assert out.free_energy == pytest.approx(E0, abs=1e-8)
    np.testing.assert_allclose(out[-1].geometry.lattice_vectors, np.loadtxt(
        io.StringIO(GEOMETRY), usecols=(1, 2, 3), max_rows=3))
    assert out.results["input"]["control_in"].splitlines()[0].strip() == "xc                 pbe"
    assert (calc / "fd" / "reference" / "control.in").read_text() == CONTROL
    strained = (calc / "fd" / "stress_xy_plus" / "control.in").read_text()
    assert "\ncompute_forces" not in strained
    assert "# aims-fd removed: compute_analytical_stress .true." in strained
    assert (calc / "fd" / "force_a0002_z_plus" / "geometry.in").read_text().endswith("Si\n")
    assert main([]) == 0
    assert len(calls(calc)) == 25
    np.testing.assert_allclose(parse(calc, capsys.readouterr().out).stress, SIGMA, atol=1e-9)


def test_forces_only_and_unfinished_runs(calc, capsys, monkeypatch):
    (calc / "control.in").write_text(CONTROL.replace("compute_analytical_stress", "# "))
    monkeypatch.setenv("FAKE_AIMS_FAIL", "force_a0002_y_minus")
    assert main([]) == 1
    assert "not finished or not converged: force_a0002_y_minus" in capsys.readouterr().err
    monkeypatch.delenv("FAKE_AIMS_FAIL")
    assert main([]) == 0
    assert calls(calc).count("force_a0002_y_minus") == 2
    assert len(calls(calc)) == 14
    out = parse(calc, capsys.readouterr().out)
    np.testing.assert_allclose(out.forces, FORCES, atol=1e-6)
    assert out.stress is None


def test_dry_then_run_elsewhere_then_evaluate(calc, capsys):
    assert main(["--dry"]) == 0
    assert capsys.readouterr().out == ""
    for run in sorted((calc / "fd").iterdir()):  # as a job farm would
        if run.is_dir():
            subprocess.run(shlex.split(FAKE), cwd=run, check=True,
                           stdout=(run / "aims.out").open("w"))
    assert main(["--dry"]) == 0  # still only writes: no evaluation
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "25 runs in fd/, 0 not finished" in captured.err
    assert len(calls(calc)) == 25  # the job farm's runs
    (calc / ".aimsfdrc").unlink()  # nothing left to run: no FHI-aims command needed
    assert main([]) == 0
    np.testing.assert_allclose(parse(calc, capsys.readouterr().out).stress, SIGMA, atol=1e-9)
    assert len(calls(calc)) == 25


def test_only_forces_or_only_stress(calc, capsys):
    assert main(["--stress"]) == 0
    out = parse(calc, capsys.readouterr().out)
    np.testing.assert_allclose(out.stress, SIGMA, atol=1e-9)
    assert out.forces is None
    assert len(calls(calc)) == 13
    assert main(["--forces"]) == 0
    out = parse(calc, capsys.readouterr().out)
    np.testing.assert_allclose(out.forces, FORCES, atol=1e-6)
    assert out.stress is None
    assert len(calls(calc)) == 25
    assert calls(calc).count("reference") == 1
    assert main(["--forces", "--stress"]) == 0
    out = parse(calc, capsys.readouterr().out)
    np.testing.assert_allclose(out.stress, SIGMA, atol=1e-9)
    np.testing.assert_allclose(out.forces, FORCES, atol=1e-6)
    assert len(calls(calc)) == 25


def contributions(text, title, n_rows):
    block = text.split(f"Contributions to the finite-difference {title}")[1].split("\n\n")[0]
    lines = block.splitlines()[1:-1]
    labels = [line.strip().rstrip(":") for line in lines[:: n_rows + 1]]
    values = np.array([[line.split()[-3:] for line in lines[k + 1:k + 1 + n_rows]]
                       for k in range(0, len(lines), n_rows + 1)], dtype=float)
    return labels, values, float(block.splitlines()[-1].split()[-2])


def test_decomposition_adds_up(calc, capsys, monkeypatch):
    monkeypatch.setenv("FAKE_AIMS_VDW", "1")
    assert main([]) == 0
    text = capsys.readouterr().out
    out = parse(calc, text)
    np.testing.assert_allclose(out.forces, FORCES + FORCES_VDW, atol=1e-6)
    np.testing.assert_allclose(out.stress, SIGMA + SIGMA_VDW, atol=1e-9)
    labels, forces, forces_sum = contributions(text, "forces", 2)
    assert labels == ["vdW energy correction", "rest of the electronic free energy"]
    # the vdW energy is printed with 8 decimals: 1e-8 eV / (2 x 0.002 A) = 2.5e-6 eV/A
    np.testing.assert_allclose(forces[0], FORCES_VDW, atol=3e-6)
    np.testing.assert_allclose(forces[1], FORCES, atol=3e-6)
    # printed totals and contributions add up to the printed digits
    np.testing.assert_allclose(forces.sum(axis=0), out.forces, rtol=0, atol=1e-13)
    assert forces_sum < 1e-15
    labels, stress, stress_sum = contributions(text, "stress", 3)
    np.testing.assert_allclose(stress[0], SIGMA_VDW, atol=2e-6)
    np.testing.assert_allclose(stress[1], SIGMA, atol=2e-6)
    np.testing.assert_allclose(stress.sum(axis=0), out.stress, rtol=0, atol=3e-10)
    assert stress_sum < 1e-15
    half = EnergyComponent("half of the vdW energy",
                           lambda output: 0.5 * pyfhiaims_value("vdw_correction")(output))
    names = plan(2, True, False, 0.002, 1e-4)[0]
    energies, _, _ = collect(calc / "fd", names, components=(half, half))
    table, labels = decomposition(energies, components=(half, half))
    assert labels == ("half of the vdW energy", "half of the vdW energy",
                      "rest of the electronic free energy")
    parts = with_rest(fd_forces(table[run_ranges(2, True, False)[0]], 2, 0.002))
    np.testing.assert_allclose(parts[1] + parts[2], forces[0], atol=3e-6)
    np.testing.assert_allclose(parts[1:].sum(axis=0), parts[0], rtol=0, atol=1e-15)


def test_no_decomposition_without_components(calc, capsys):
    assert main(["--forces"]) == 0
    assert "Contributions" not in capsys.readouterr().out


@pytest.mark.parametrize("flags", [[], ["--forces"], ["--stress"]])
@pytest.mark.parametrize("vdw", [False, True])
def test_ase_reads_the_output(calc, capsys, monkeypatch, flags, vdw):
    ase_io = pytest.importorskip("ase.io")
    if vdw:
        monkeypatch.setenv("FAKE_AIMS_VDW", "1")
    assert main(flags) == 0
    text = capsys.readouterr().out
    out = parse(calc, text)
    atoms = ase_io.read(calc / "aims.out", format="aims-output")
    assert atoms.get_chemical_formula() == "Si2"
    np.testing.assert_allclose(atoms.cell[:], out[-1].geometry.lattice_vectors, atol=1e-8)
    assert atoms.get_potential_energy(force_consistent=True) == out.free_energy
    if out.forces is not None:
        np.testing.assert_array_equal(atoms.get_forces(), np.array(out.forces))
    if out.stress is not None:
        np.testing.assert_array_equal(atoms.get_stress(voigt=False), np.array(out.stress))


def test_version(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith(f"aims-fd {aims_fd.__version__} (pyfhiaims ")


@pytest.mark.parametrize("line", ["output band 0 0 0 0.5 0 0 20 G X",
                                  "relax_geometry bfgs 1e-2",
                                  "MD_time_step 0.001",
                                  "total_energy_method rpa",
                                  "DFPT dielectric"])
def test_unsupported_calculations_stop(calc, capsys, line):
    (calc / "control.in").write_text(CONTROL.replace("k_grid", f"{line}\nk_grid"))
    assert main([]) == 1
    error = capsys.readouterr().err
    assert f"line 3: {line}" in error  # inserted before k_grid, the third line
    assert not (calc / "fd").exists()
    assert calls(calc) == []


def test_errors(calc, capsys):
    assert main(["--dry"]) == 0
    assert "25 runs in fd/, 25 not finished" in capsys.readouterr().err
    assert (calc / "fd" / "stress_xy_minus" / "geometry.in").is_file()
    assert calls(calc) == []
    assert main(["--delta", "0.001"]) == 1
    assert "made from other inputs or steps" in capsys.readouterr().err
    (calc / "control.in").write_text("xc pbe\nsc_accuracy_rho 1e-7\n" + SPECIES)
    assert main(["--dir", "other"]) == 1
    assert "asks for neither" in capsys.readouterr().err
    (calc / ".aimsfdrc").unlink()
    (calc / "control.in").write_text(CONTROL)
    assert main(["--dir", "third"]) == 1
    assert "no FHI-aims command" in capsys.readouterr().err

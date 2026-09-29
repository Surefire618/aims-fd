"""Tests of aims-fd with tests/fake_aims.py in place of FHI-aims."""

import io
import pathlib
import shlex
import subprocess
import sys

import numpy as np
import pyfhiaims
import pytest
from pyfhiaims.outputs import stdout as aims_stdout

import aims_fd
from aims_fd import cli
from aims_fd import control
from aims_fd import deformations
from aims_fd import energies
from aims_fd import finite_differences
from tests import fake_aims

FAKE = (
    f"{shlex.quote(sys.executable)} "
    f"{shlex.quote(str(pathlib.Path(__file__).parent / 'fake_aims.py'))}"
)
SPECIES = (
    "\n  species        Si\n    nucleus             14\n"
    "    mass                28.0855\n"
)
CONTROL = (
    "xc                 pbe\n"
    "occupation_type    gaussian 0.01\n"
    "k_grid             10 10 10\n"
    "sc_accuracy_rho    1e-7\n"
    "sc_accuracy_etot   1e-9\n"
    "compute_forces     .true.\n"
    "compute_analytical_stress .true.\n" + SPECIES
)
GEOMETRY = (
    "lattice_vector 0.0 2.7155 2.7155\n"
    "lattice_vector 2.7155 0.0 2.7155\n"
    "lattice_vector 2.7155 2.7155 0.0\n"
    "atom 0.0 0.0 0.0 Si\n"
    "atom 1.424186383883 1.397611830330 1.377680915165 Si\n"
)
REST = "rest of the electronic free energy"


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
    return aims_stdout.AimsStdout(calc / "aims.out")


def calls(calc):
    log = calc / "calls.log"
    return log.read_text().split() if log.exists() else []


def test_requests_and_comment_out():
    assert control.requests(CONTROL) == (True, True, 1e-4)
    assert control.requests(
        "compute_forces .false.\ncompute_numerical_stress T\n"
        "delta_numerical_stress 2d-4\n"
    ) == (False, True, 2e-4)
    assert control.requests("# compute_forces .true.\n") == (False, False, 1e-4)
    edited = control.comment_out(
        CONTROL, ("compute_forces", "compute_analytical_stress")
    )
    assert "# aims-fd removed: compute_forces     .true." in edited
    assert edited.count("\n") == CONTROL.count("\n")
    assert all(
        line.startswith("#") or "compute" not in line
        for line in edited.splitlines()
    )


def test_aims_fd_writes_an_aims_out_that_pyfhiaims_reads(calc, capsys):
    assert cli.main([]) == 0
    # reference + 2 atoms x 3 axes x 2 signs + 6 components x 2 signs
    assert len(calls(calc)) == 25
    text = capsys.readouterr().out
    out = parse(calc, text)
    assert f"  aims-fd version       : {aims_fd.__version__}\n" in text
    assert f"  pyfhiaims version     : {pyfhiaims.__version__}\n" in text
    assert out.aims_version == "260824"
    assert out.is_finished_ok
    assert out.converged
    np.testing.assert_allclose(out.forces, fake_aims.FORCES, atol=1e-6)
    np.testing.assert_allclose(out.stress, fake_aims.SIGMA, atol=1e-9)
    assert out.free_energy == pytest.approx(fake_aims.E0, abs=1e-8)
    lattice = np.loadtxt(io.StringIO(GEOMETRY), usecols=(1, 2, 3), max_rows=3)
    np.testing.assert_allclose(out[-1].geometry.lattice_vectors, lattice)
    first_line = out.results["input"]["control_in"].splitlines()[0]
    assert first_line.strip() == "xc                 pbe"
    assert (calc / "fd" / "reference" / "control.in").read_text() == CONTROL
    strained = (calc / "fd" / "stress_xy_plus" / "control.in").read_text()
    assert "\ncompute_forces" not in strained
    assert "# aims-fd removed: compute_analytical_stress .true." in strained
    displaced = calc / "fd" / "force_a0002_z_plus" / "geometry.in"
    assert displaced.read_text().endswith("Si\n")
    assert cli.main([]) == 0
    assert len(calls(calc)) == 25
    out = parse(calc, capsys.readouterr().out)
    np.testing.assert_allclose(out.stress, fake_aims.SIGMA, atol=1e-9)


def test_forces_only_and_unfinished_runs(calc, capsys, monkeypatch):
    (calc / "control.in").write_text(
        CONTROL.replace("compute_analytical_stress", "# ")
    )
    monkeypatch.setenv("FAKE_AIMS_FAIL", "force_a0002_y_minus")
    assert cli.main([]) == 1
    error = capsys.readouterr().err
    assert "not finished or not converged: force_a0002_y_minus" in error
    monkeypatch.delenv("FAKE_AIMS_FAIL")
    assert cli.main([]) == 0
    assert calls(calc).count("force_a0002_y_minus") == 2
    assert len(calls(calc)) == 14
    out = parse(calc, capsys.readouterr().out)
    np.testing.assert_allclose(out.forces, fake_aims.FORCES, atol=1e-6)
    assert out.stress is None


def test_dry_then_run_elsewhere_then_evaluate(calc, capsys):
    assert cli.main(["--dry"]) == 0
    assert capsys.readouterr().out == ""
    for run in sorted((calc / "fd").iterdir()):  # As a job farm would.
        if run.is_dir():
            with (run / "aims.out").open("w") as output:
                subprocess.run(
                    shlex.split(FAKE), cwd=run, check=True, stdout=output
                )
    assert cli.main(["--dry"]) == 0  # Still only writes: no evaluation.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "25 runs in fd/, 0 not finished" in captured.err
    assert len(calls(calc)) == 25  # The job farm's runs.
    # Nothing left to run: no FHI-aims command needed.
    (calc / ".aimsfdrc").unlink()
    assert cli.main([]) == 0
    out = parse(calc, capsys.readouterr().out)
    np.testing.assert_allclose(out.stress, fake_aims.SIGMA, atol=1e-9)
    assert len(calls(calc)) == 25


def test_only_forces_or_only_stress(calc, capsys):
    assert cli.main(["--stress"]) == 0
    out = parse(calc, capsys.readouterr().out)
    np.testing.assert_allclose(out.stress, fake_aims.SIGMA, atol=1e-9)
    assert out.forces is None
    assert len(calls(calc)) == 13
    assert cli.main(["--forces"]) == 0
    out = parse(calc, capsys.readouterr().out)
    np.testing.assert_allclose(out.forces, fake_aims.FORCES, atol=1e-6)
    assert out.stress is None
    assert len(calls(calc)) == 25
    assert calls(calc).count("reference") == 1
    assert cli.main(["--forces", "--stress"]) == 0
    out = parse(calc, capsys.readouterr().out)
    np.testing.assert_allclose(out.stress, fake_aims.SIGMA, atol=1e-9)
    np.testing.assert_allclose(out.forces, fake_aims.FORCES, atol=1e-6)
    assert len(calls(calc)) == 25


def contributions(text, title, n_rows):
    """Reads a contributions block: labels, values and the printed sum."""
    block = text.split(f"Contributions to the finite-difference {title}")[1]
    block = block.split("\n\n")[0]
    lines = block.splitlines()[1:-1]
    labels = [line.strip().rstrip(":") for line in lines[:: n_rows + 1]]
    values = np.array(
        [
            [line.split()[-3:] for line in lines[k + 1 : k + 1 + n_rows]]
            for k in range(0, len(lines), n_rows + 1)
        ],
        dtype=float,
    )
    return labels, values, float(block.splitlines()[-1].split()[-2])


def test_decomposition_adds_up(calc, capsys, monkeypatch):
    monkeypatch.setenv("FAKE_AIMS_VDW", "1")
    assert cli.main([]) == 0
    text = capsys.readouterr().out
    out = parse(calc, text)
    np.testing.assert_allclose(
        out.forces, fake_aims.FORCES + fake_aims.FORCES_VDW, atol=1e-6
    )
    np.testing.assert_allclose(
        out.stress, fake_aims.SIGMA + fake_aims.SIGMA_VDW, atol=1e-9
    )
    labels, forces, forces_sum = contributions(text, "forces", 2)
    assert labels == ["vdW energy correction", REST]
    # The vdW energy is printed with 8 decimals:
    # 1e-8 eV / (2 x 0.002 A) = 2.5e-6 eV/A.
    np.testing.assert_allclose(forces[0], fake_aims.FORCES_VDW, atol=3e-6)
    np.testing.assert_allclose(forces[1], fake_aims.FORCES, atol=3e-6)
    # The printed totals and contributions add up to the printed digits.
    np.testing.assert_allclose(
        forces.sum(axis=0), out.forces, rtol=0, atol=1e-13
    )
    assert forces_sum < 1e-15
    labels, stress, stress_sum = contributions(text, "stress", 3)
    np.testing.assert_allclose(stress[0], fake_aims.SIGMA_VDW, atol=2e-6)
    np.testing.assert_allclose(stress[1], fake_aims.SIGMA, atol=2e-6)
    np.testing.assert_allclose(
        stress.sum(axis=0), out.stress, rtol=0, atol=3e-10
    )
    assert stress_sum < 1e-15
    vdw = energies.pyfhiaims_value("vdw_correction")
    half = energies.EnergyComponent(
        "half of the vdW energy", lambda output: 0.5 * vdw(output)
    )
    names = deformations.plan(2, True, False, 0.002, 1e-4)[0]
    run_energies, _, _ = energies.collect(
        calc / "fd", names, components=(half, half)
    )
    table, labels = energies.decomposition(
        run_energies, components=(half, half)
    )
    assert labels == ("half of the vdW energy", "half of the vdW energy", REST)
    force_runs = deformations.run_ranges(2, True, False)[0]
    parts = energies.with_rest(
        finite_differences.fd_forces(table[force_runs], 2, 0.002)
    )
    np.testing.assert_allclose(parts[1] + parts[2], forces[0], atol=3e-6)
    np.testing.assert_allclose(
        parts[1:].sum(axis=0), parts[0], rtol=0, atol=1e-15
    )


def test_no_decomposition_without_components(calc, capsys):
    assert cli.main(["--forces"]) == 0
    assert "Contributions" not in capsys.readouterr().out


@pytest.mark.parametrize("flags", [[], ["--forces"], ["--stress"]])
@pytest.mark.parametrize("vdw", [False, True])
def test_ase_reads_the_output(calc, capsys, monkeypatch, flags, vdw):
    ase_io = pytest.importorskip("ase.io")
    if vdw:
        monkeypatch.setenv("FAKE_AIMS_VDW", "1")
    assert cli.main(flags) == 0
    text = capsys.readouterr().out
    out = parse(calc, text)
    atoms = ase_io.read(calc / "aims.out", format="aims-output")
    assert atoms.get_chemical_formula() == "Si2"
    np.testing.assert_allclose(
        atoms.cell[:], out[-1].geometry.lattice_vectors, atol=1e-8
    )
    assert atoms.get_potential_energy(force_consistent=True) == out.free_energy
    if out.forces is not None:
        np.testing.assert_array_equal(atoms.get_forces(), np.array(out.forces))
    if out.stress is not None:
        np.testing.assert_array_equal(
            atoms.get_stress(voigt=False), np.array(out.stress)
        )


def test_version(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith(
        f"aims-fd {aims_fd.__version__} (pyfhiaims "
    )


@pytest.mark.parametrize(
    "line",
    [
        "output band 0 0 0 0.5 0 0 20 G X",
        "relax_geometry bfgs 1e-2",
        "MD_time_step 0.001",
        "total_energy_method rpa",
        "DFPT dielectric",
    ],
)
def test_unsupported_calculations_stop(calc, capsys, line):
    (calc / "control.in").write_text(
        CONTROL.replace("k_grid", f"{line}\nk_grid")
    )
    assert cli.main([]) == 1
    error = capsys.readouterr().err
    assert f"line 3: {line}" in error  # Inserted before k_grid, line 3.
    assert not (calc / "fd").exists()
    assert calls(calc) == []


def test_errors(calc, capsys):
    assert cli.main(["--dry"]) == 0
    assert "25 runs in fd/, 25 not finished" in capsys.readouterr().err
    assert (calc / "fd" / "stress_xy_minus" / "geometry.in").is_file()
    assert calls(calc) == []
    assert cli.main(["--delta", "0.001"]) == 1
    assert "made from other inputs or steps" in capsys.readouterr().err
    (calc / "control.in").write_text("xc pbe\nsc_accuracy_rho 1e-7\n" + SPECIES)
    assert cli.main(["--dir", "other"]) == 1
    assert "asks for neither" in capsys.readouterr().err
    (calc / ".aimsfdrc").unlink()
    (calc / "control.in").write_text(CONTROL)
    assert cli.main(["--dir", "third"]) == 1
    assert "no FHI-aims command" in capsys.readouterr().err

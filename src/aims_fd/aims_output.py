"""The FD result as an aims.out.

The sections of templates/aims.out.template follow the order in which the pyfhiaims parser
expects them. The header lines that ASE requires are taken from pyfhiaims' reading of the
reference run and printed in FHI-aims' format.
"""


from importlib import resources
from string import Template

import numpy as np
import pyfhiaims
from pyfhiaims.geometry import AimsGeometry
from pyfhiaims.outputs.stdout import AimsStdout

from .control import last_value
from .deformations import AXES
from .energies import closure
from .version import __version__

TEMPLATES = resources.files("aims_fd") / "templates"
FORCES_TITLE = ("  Total atomic forces (finite differences of the electronic free energy)"
                " [eV/Ang]:\n")


def _template(name: str) -> Template:
    return Template((TEMPLATES / name).read_text())


def _verbatim(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def _force_rows(forces: np.ndarray) -> str:
    return "".join(f"  | {atom:4d}   {f[0]:23.15E}   {f[1]:23.15E}   {f[2]:23.15E}\n"
                   for atom, f in enumerate(forces, 1))


def _stress_rows(stress: np.ndarray) -> list[str]:
    return [f"{row[0]:17.10f}{row[1]:17.10f}{row[2]:17.10f}" for row in stress]


# The titles differ from those that pyfhiaims and ASE parse, so both keep reading the totals.
def _contributions(title: str, unit: str, labels: tuple[str, ...], rows: list[str],
                   values: np.ndarray) -> str:
    return (f"  Contributions to the finite-difference {title} [{unit}]:\n"
            + "".join(f"    {label}:\n{row}" for label, row in zip(labels, rows, strict=True))
            + f"    sum of the contributions - total: max |.| = {closure(values):.1E} {unit}\n\n")


def _input_geometry(geometry_text: str) -> str:
    """The "Input geometry" block; ASE reads the structure from it."""
    geometry = AimsGeometry.from_strings(geometry_text.splitlines())
    lines = []
    if geometry.lattice_vectors is not None:
        lines.append("  | Unit cell:")
        lines += [f"  |{vector[0]:18.8f}{vector[1]:18.8f}{vector[2]:18.8f}"
                  for vector in np.asarray(geometry.lattice_vectors, float)]
    lines += ["  | Atomic structure:",
              "  |       Atom                x [A]            y [A]            z [A]"]
    for index, atom in enumerate(geometry.atoms, 1):
        x, y, z = np.asarray(atom.position, float)
        lines.append(f"  |{index:5d}: Species {atom.symbol:<2s}{x:22.8f}{y:18.8f}{z:18.8f}")
    return "\n".join(lines)


def _fortran_e(value: float, digits: int = 6) -> str:
    """Fortran E format, e.g. 0.100000E-01."""
    if value == 0:
        return f"0.{'0' * digits}E+00"
    exponent = int(np.floor(np.log10(abs(value)))) + 1
    mantissa = f"{value / 10.0**exponent:.{digits}f}"
    if mantissa.lstrip("-").startswith("1"):
        exponent += 1
        mantissa = f"{value / 10.0**exponent:.{digits}f}"
    return f"{mantissa}E{exponent:+03d}"


def _line(text: str, value) -> str:
    return text.format(value) + "\n" if value is not None else ""


def _header_facts(control: str, geometry_text: str, reference: AimsStdout) -> tuple[str, ...]:
    meta = reference.metadata
    width = reference.results["input"].get("electronic_temperature")
    occupation = (last_value(control, "occupation_type") or ["gaussian"])[0].capitalize()
    n_atoms = len(AimsGeometry.from_strings(geometry_text.splitlines()).atoms)
    return (
        _line("  | Number of spin channels           : {:8d}", meta.get("num_spins")),
        _line(f"  Occupation type: {occupation} broadening, width = {{:>14s}} eV.",
              None if width is None else _fortran_e(width)),
        _line(f"  The structure contains {n_atoms:8d} atoms,  and a total of {{:14.3f}} "
              "electrons.", meta.get("num_electrons")),
        _line("  | Number of Kohn-Sham states (occupied + empty): {:8d}", meta.get("num_bands")),
        _line("  | Number of k-points                             : {:9d}",
              meta.get("num_k_points")),
    )


def aims_out(control: str, geometry_text: str, reference: AimsStdout, times: np.ndarray,
             forces: np.ndarray | None, stress: np.ndarray | None, labels: tuple[str, ...],
             delta: float, strain_step: float) -> str:
    """forces (n_rows x n_atoms x 3) and stress (n_rows x 3 x 3) hold the total in row 0 and the
    contributions named by labels in the other rows."""
    final = reference.results["ionic_steps"][-1]
    facts = _header_facts(control, geometry_text, reference)
    cpu, wall = times.sum(axis=0)
    results = ""
    if forces is not None:
        results += FORCES_TITLE + _force_rows(forces[0]) + "\n"
    if stress is not None:
        results += _template("stress.template").substitute(
            dict(zip(AXES, _stress_rows(stress[0]), strict=True)),
            pressure=f"{-np.trace(stress[0]) / 3:18.10f}")
    if forces is not None and len(forces) > 1:
        results += _contributions("forces", "eV/Ang", labels,
                                  [_force_rows(part) for part in forces[1:]], forces)
    if stress is not None and len(stress) > 1:
        results += _contributions("stress", "eV/A**3", labels, [
            "".join(f"  |  {axis}  {row}\n" for axis, row in zip(AXES, _stress_rows(part),
                                                               strict=True))
            for part in stress[1:]], stress)
    return _template("aims.out.template").substitute(
        aims_fd_version=__version__, pyfhiaims_version=pyfhiaims.__version__,
        n_runs=len(times), delta=delta, strain_step=strain_step,
        aims_version=reference.aims_version, commit_hash=reference.metadata.get("commit_hash"),
        control_in=_verbatim(control), geometry_in=_verbatim(geometry_text),
        input_geometry=_input_geometry(geometry_text),
        spin_channels=facts[0], occupation_type=facts[1], electrons=facts[2],
        kohn_sham_states=facts[3], k_points=facts[4],
        n_atoms=f"{len(AimsGeometry.from_strings(geometry_text.splitlines()).atoms):8d}",
        total_energy=f"{final['total_energy']:23.15E}",
        corrected_total_energy=f"{final['corrected_total_energy']:23.15E}",
        free_energy=f"{final['free_energy']:23.15E}",
        results=results, cpu_time=f"{cpu:12.3f}", wall_time=f"{wall:12.3f}",
        memory_min=f"{reference.memory_min_mb or 0.0:12.3f}",
        memory_max=f"{reference.memory_max_mb or 0.0:12.3f}",
        memory_avg=f"{reference.memory_avg_mb or 0.0:12.3f}")

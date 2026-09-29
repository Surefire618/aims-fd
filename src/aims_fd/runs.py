"""Writing and running the FHI-aims calculations of a finite-difference set."""

import hashlib
import pathlib
import subprocess
import sys

import numpy as np
from pyfhiaims import geometry as aims_geometry

from aims_fd import control as control_lib
from aims_fd import deformations
from aims_fd import errors

SETTINGS_NAME = "aims-fd.settings"


def _check_inputs(control: str, forces: bool, stress: bool) -> None:
    lines = control_lib.unsupported(control)
    if lines:
        raise errors.FDError(
            "control.in asks for calculations that aims-fd does not support; "
            "they would be repeated in every run, or they change the geometry "
            "or the energy that aims-fd differentiates. Remove these lines:\n  "
            + "\n  ".join(lines)
        )
    if not (forces or stress):
        raise errors.FDError(
            "nothing to compute: control.in asks for neither compute_forces "
            "nor compute_analytical_stress / compute_numerical_stress .true. "
            "(or use --forces / --stress)"
        )


def _settings(
    control: str, geometry_text: str, delta: float, strain_step: float
) -> str:
    control_hash = hashlib.sha256(control.encode()).hexdigest()
    geometry_hash = hashlib.sha256(geometry_text.encode()).hexdigest()
    return (
        f"delta {delta!r} strain_step {strain_step!r} "
        f"control.in {control_hash} geometry.in {geometry_hash}\n"
    )


def write_runs(
    root: pathlib.Path,
    control: str,
    geometry_text: str,
    forces: bool,
    stress: bool,
    delta: float,
    strain_step: float,
) -> tuple[aims_geometry.AimsGeometry, np.ndarray]:
    """Writes the run directories, except those that have an aims.out.

    Args:
        root: Directory of the runs.
        control: Text of control.in.
        geometry_text: Text of geometry.in.
        forces: Whether to write the force runs.
        stress: Whether to write the stress runs.
        delta: Atom displacement in Angstrom.
        strain_step: Strain step h.

    Returns:
        The input geometry and the run names.

    Raises:
        FDError: If the inputs cannot be used, or if root holds runs made
            from other inputs or steps.
    """
    _check_inputs(control, forces, stress)
    geometry = aims_geometry.AimsGeometry.from_strings(
        geometry_text.splitlines()
    )
    if any(atom.is_empty or atom.is_pseudocore for atom in geometry.atoms):
        raise errors.FDError("empty and pseudocore sites are not supported")
    if stress and geometry.lattice_vectors is None:
        raise errors.FDError(
            "the stress needs a periodic geometry (lattice_vector lines)"
        )
    # Runs made from other inputs or steps must not be mixed into one set.
    settings = _settings(control, geometry_text, delta, strain_step)
    saved = root / SETTINGS_NAME
    if saved.exists() and saved.read_text() != settings:
        raise errors.FDError(
            f"{root} was made from other inputs or steps; remove it or "
            "choose another directory"
        )
    names, displacements, deformation_matrices = deformations.plan(
        len(geometry.atoms), forces, stress, delta, strain_step
    )
    root.mkdir(parents=True, exist_ok=True)
    saved.write_text(settings)
    for run, name in enumerate(names):
        run_dir = root / name
        if (run_dir / "aims.out").exists():
            continue
        run_dir.mkdir(exist_ok=True)
        if run == 0:
            removed = control_lib.REMOVED_IN_REFERENCE
        else:
            removed = control_lib.REMOVED
        (run_dir / "control.in").write_text(
            control_lib.comment_out(control, removed)
        )
        text = deformations.run_geometry(
            geometry, displacements[run], deformation_matrices[run]
        ).to_string()
        # pyfhiaims omits the final newline.
        (run_dir / "geometry.in").write_text(
            f"# aims-fd run {name}\n{text.rstrip()}\n"
        )
    return geometry, names


def finished(output: pathlib.Path) -> bool:
    if not output.exists():
        return False
    return "Have a nice day." in output.read_text(errors="replace")[-3000:]


def run_all(root: pathlib.Path, names: np.ndarray, command: str) -> None:
    """Runs FHI-aims, one run after another, where aims.out is not finished."""
    for run, name in enumerate(names):
        run_dir = root / name
        if finished(run_dir / "aims.out"):
            continue
        print(f"[{run + 1}/{len(names)}] {name}", file=sys.stderr, flush=True)
        with (
            open(run_dir / "aims.out", "w") as out,
            open(run_dir / "aims.err", "w") as err,
        ):
            subprocess.run(
                ["bash", "-c", command],
                cwd=run_dir,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                check=False,
            )

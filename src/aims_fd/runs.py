import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from pyfhiaims.geometry import AimsGeometry

from .control import REMOVED, REMOVED_IN_REFERENCE, comment_out, unsupported
from .deformations import plan, run_geometry
from .errors import FDError

SETTINGS_NAME = "aims-fd.settings"


def write_runs(root: Path, control: str, geometry_text: str, forces: bool, stress: bool,
               delta: float, strain_step: float) -> tuple[AimsGeometry, np.ndarray]:
    """Write the run directories below root, except those that already have an aims.out.

    Returns the input geometry and the run names.
    """
    lines = unsupported(control)
    if lines:
        raise FDError("control.in asks for calculations that aims-fd does not support; they would "
                      "be repeated in every run, or they change the geometry or the energy that "
                      "aims-fd differentiates. Remove these lines:\n  " + "\n  ".join(lines))
    if not (forces or stress):
        raise FDError("nothing to compute: control.in asks for neither compute_forces nor "
                      "compute_analytical_stress / compute_numerical_stress .true. (or use "
                      "--forces / --stress)")
    geometry = AimsGeometry.from_strings(geometry_text.splitlines())
    if any(atom.is_empty or atom.is_pseudocore for atom in geometry.atoms):
        raise FDError("empty and pseudocore sites are not supported")
    if stress and geometry.lattice_vectors is None:
        raise FDError("the stress needs a periodic geometry (lattice_vector lines)")
    settings = (f"delta {delta!r} strain_step {strain_step!r} control.in "
                f"{hashlib.sha256(control.encode()).hexdigest()} geometry.in "
                f"{hashlib.sha256(geometry_text.encode()).hexdigest()}\n")
    # runs made from other inputs or steps must not be mixed into one set
    saved = root / SETTINGS_NAME
    if saved.exists() and saved.read_text() != settings:
        raise FDError(f"{root} was made from other inputs or steps; remove it or choose "
                      "another directory")
    names, displacements, deformations = plan(len(geometry.atoms), forces, stress, delta,
                                              strain_step)
    root.mkdir(parents=True, exist_ok=True)
    saved.write_text(settings)
    for run, name in enumerate(names):
        run_dir = root / name
        if (run_dir / "aims.out").exists():
            continue
        run_dir.mkdir(exist_ok=True)
        removed = REMOVED_IN_REFERENCE if run == 0 else REMOVED
        (run_dir / "control.in").write_text(comment_out(control, removed))
        text = run_geometry(geometry, displacements[run], deformations[run]).to_string()
        # pyfhiaims omits the final newline
        (run_dir / "geometry.in").write_text(f"# aims-fd run {name}\n{text.rstrip()}\n")
    return geometry, names


def finished(output: Path) -> bool:
    return output.exists() and "Have a nice day." in output.read_text(errors="replace")[-3000:]


def run_all(root: Path, names: np.ndarray, command: str) -> None:
    for run, name in enumerate(names):
        run_dir = root / name
        if finished(run_dir / "aims.out"):
            continue
        print(f"[{run + 1}/{len(names)}] {name}", file=sys.stderr, flush=True)
        with open(run_dir / "aims.out", "w") as out, open(run_dir / "aims.err", "w") as err:
            subprocess.run(["bash", "-c", command], cwd=run_dir, stdin=subprocess.DEVNULL,
                           stdout=out, stderr=err, check=False)

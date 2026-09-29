import argparse
import configparser
import sys
from pathlib import Path

import numpy as np
import pyfhiaims

from .aims_output import aims_out
from .control import requests
from .deformations import run_ranges
from .energies import collect, decomposition, with_rest
from .errors import FDError
from .finite_differences import fd_forces, fd_stress
from .runs import finished, run_all, write_runs
from .version import __version__


def aims_command(given: str | None) -> str:
    """--aims-command, else [machine] aims_command of ./.aimsfdrc, else of ~/.aimsfdrc."""
    if given:
        return given
    rc = configparser.ConfigParser(interpolation=None)
    rc.read([Path.home() / ".aimsfdrc", Path(".aimsfdrc")])
    command = rc.get("machine", "aims_command", fallback="").strip()
    if not command:
        raise FDError("no FHI-aims command: put '[machine]' and 'aims_command = mpirun ... "
                      "aims.x' into ~/.aimsfdrc or ./.aimsfdrc, or use --aims-command")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aims-fd",
        description="Finite-difference forces and stress for FHI-aims. Run it like aims.x in a "
        "directory with control.in and geometry.in: aims-fd > aims.out")
    parser.add_argument("--version", action="version",
                        version=f"aims-fd {__version__} (pyfhiaims {pyfhiaims.__version__})")
    parser.add_argument("--forces", action="store_true",
                        help="compute the forces (with --stress: both); default: what "
                        "control.in asks for")
    parser.add_argument("--stress", action="store_true",
                        help="compute the stress (with --forces: both); default: what "
                        "control.in asks for")
    parser.add_argument("--delta", type=float, default=0.002,
                        help="atom displacement in Angstrom (default: %(default)s)")
    parser.add_argument("--strain-step", type=float,
                        help="strain step (default: delta_numerical_stress of control.in, "
                        "else 1e-4)")
    parser.add_argument("--dir", default="fd", help="directory of the runs (default: fd)")
    parser.add_argument("--aims-command", help="command that runs FHI-aims (overrides "
                        ".aimsfdrc)")
    parser.add_argument("--dry", action="store_true",
                        help="only write the runs, to run them elsewhere (e.g. with a job farm); "
                        "afterwards, aims-fd without --dry evaluates them")
    args = parser.parse_args(argv)
    try:
        control = Path("control.in").read_text()
        geometry_text = Path("geometry.in").read_text()
        forces, stress, strain_step = requests(control)
        if args.forces or args.stress:
            forces, stress = args.forces, args.stress
        strain_step = args.strain_step or strain_step
        root = Path(args.dir)
        geometry, names = write_runs(root, control, geometry_text, forces, stress, args.delta,
                                     strain_step)
        n_pending = sum(not finished(root / name / "aims.out") for name in names)
        if args.dry:
            print(f"aims-fd --dry: {len(names)} runs in {root}/, {n_pending} not finished",
                  file=sys.stderr)
            return 0
        if n_pending:
            run_all(root, names, aims_command(args.aims_command))
        energies, times, reference = collect(root, names)
    except (FDError, OSError, ValueError) as exc:
        print(f"aims-fd: {exc}", file=sys.stderr)
        return 1
    table, labels = decomposition(energies)
    force_runs, stress_runs = run_ranges(len(geometry.atoms), forces, stress)
    forces_rows = stress_rows = None
    if forces:
        forces_rows = with_rest(fd_forces(table[force_runs], len(geometry.atoms),
                                          args.delta))
    if stress:
        volume = abs(np.linalg.det(np.asarray(geometry.lattice_vectors, float)))
        stress_rows = with_rest(fd_stress(table[stress_runs], strain_step, volume))
    sys.stdout.write(aims_out(control, geometry_text, reference, times, forces_rows, stress_rows,
                              labels, args.delta, strain_step))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

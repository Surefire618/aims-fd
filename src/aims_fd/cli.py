"""The aims-fd command."""

import argparse
import configparser
import pathlib
import sys

import numpy as np
import pyfhiaims

from aims_fd import aims_output
from aims_fd import control as control_lib
from aims_fd import deformations
from aims_fd import energies as energies_lib
from aims_fd import errors
from aims_fd import finite_differences
from aims_fd import runs
from aims_fd import version


def aims_command(given: str | None) -> str:
    """--aims-command, else aims_command of ./.aimsfdrc or ~/.aimsfdrc."""
    if given:
        return given
    rc = configparser.ConfigParser(interpolation=None)
    rc.read([pathlib.Path.home() / ".aimsfdrc", pathlib.Path(".aimsfdrc")])
    command = rc.get("machine", "aims_command", fallback="").strip()
    if not command:
        raise errors.FDError(
            "no FHI-aims command: put '[machine]' and 'aims_command = mpirun "
            "... aims.x' into ~/.aimsfdrc or ./.aimsfdrc, or use "
            "--aims-command"
        )
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aims-fd",
        description="Finite-difference forces and stress for FHI-aims. Run "
        "it like aims.x in a directory with control.in and geometry.in: "
        "aims-fd > aims.out",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"aims-fd {version.__version__} "
        f"(pyfhiaims {pyfhiaims.__version__})",
    )
    parser.add_argument(
        "--forces",
        action="store_true",
        help="compute the forces (with --stress: both); default: what "
        "control.in asks for",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="compute the stress (with --forces: both); default: what "
        "control.in asks for",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.002,
        help="atom displacement in Angstrom (default: %(default)s)",
    )
    parser.add_argument(
        "--strain-step",
        type=float,
        help="strain step (default: delta_numerical_stress of control.in, "
        "else 1e-4)",
    )
    parser.add_argument(
        "--dir", default="fd", help="directory of the runs (default: fd)"
    )
    parser.add_argument(
        "--aims-command",
        help="command that runs FHI-aims (overrides .aimsfdrc)",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="only write the runs, to run them elsewhere (e.g. with a job "
        "farm); afterwards, aims-fd without --dry evaluates them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs aims-fd, prints the aims.out and returns the exit status."""
    args = _parser().parse_args(argv)
    try:
        control = pathlib.Path("control.in").read_text()
        geometry_text = pathlib.Path("geometry.in").read_text()
        forces, stress, strain_step = control_lib.requests(control)
        if args.forces or args.stress:
            forces, stress = args.forces, args.stress
        strain_step = args.strain_step or strain_step
        root = pathlib.Path(args.dir)
        geometry, names = runs.write_runs(
            root,
            control,
            geometry_text,
            forces,
            stress,
            args.delta,
            strain_step,
        )
        n_pending = sum(
            not runs.finished(root / name / "aims.out") for name in names
        )
        if args.dry:
            print(
                f"aims-fd --dry: {len(names)} runs in {root}/, {n_pending} "
                "not finished",
                file=sys.stderr,
            )
            return 0
        if n_pending:
            runs.run_all(root, names, aims_command(args.aims_command))
        energies, times, reference = energies_lib.collect(root, names)
    except (errors.FDError, OSError, ValueError) as exc:
        print(f"aims-fd: {exc}", file=sys.stderr)
        return 1
    table, labels = energies_lib.decomposition(energies)
    n_atoms = len(geometry.atoms)
    force_runs, stress_runs = deformations.run_ranges(n_atoms, forces, stress)
    forces_rows = stress_rows = None
    if forces:
        forces_rows = energies_lib.with_rest(
            finite_differences.fd_forces(table[force_runs], n_atoms, args.delta)
        )
    if stress:
        lattice = np.asarray(geometry.lattice_vectors, float)
        volume = abs(np.linalg.det(lattice))
        stress_rows = energies_lib.with_rest(
            finite_differences.fd_stress(
                table[stress_runs], strain_step, volume
            )
        )
    sys.stdout.write(
        aims_output.aims_out(
            control,
            geometry_text,
            reference,
            times,
            forces_rows,
            stress_rows,
            labels,
            args.delta,
            strain_step,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

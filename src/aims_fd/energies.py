"""Energies of the runs and their decomposition into components."""

import dataclasses
import pathlib
from collections.abc import Callable

import numpy as np
from pyfhiaims.outputs import stdout as aims_stdout

from aims_fd import errors


@dataclasses.dataclass(frozen=True)
class EnergyComponent:
    """An additive part of the free energy that the runs print.

    Attributes:
        label: Name printed with its contribution.
        read: Returns the component of one run in eV, or None if the run does
            not print it.
    """

    label: str
    read: Callable[[aims_stdout.AimsStdout], float | None]


def pyfhiaims_value(
    key: str,
) -> Callable[[aims_stdout.AimsStdout], float | None]:
    """Reader of a value that pyfhiaims parses into the last ionic step."""

    def read(output: aims_stdout.AimsStdout) -> float | None:
        return output.results["ionic_steps"][-1].get(key)

    return read


ENERGY_COMPONENTS = (
    EnergyComponent("vdW energy correction", pyfhiaims_value("vdw_correction")),
)
REST_LABEL = "rest of the electronic free energy"


def collect(
    root: pathlib.Path,
    names: np.ndarray,
    components: tuple[EnergyComponent, ...] = ENERGY_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, aims_stdout.AimsStdout]:
    """Reads the energies and times of all runs.

    Args:
        root: Directory of the runs.
        names: The run names.
        components: Energy components to read.

    Returns:
        The energies (n_runs x (1 + n_components), eV), the times
        (n_runs x 2: cpu, wall) and the parsed reference output. Column 0 of
        the energies is the free energy, the others are the components (NaN
        where a run does not print one).

    Raises:
        FDError: If a run did not finish or its SCF did not converge.
    """
    energies = np.full((len(names), 1 + len(components)), np.nan)
    times = np.zeros((len(names), 2))
    ok = np.zeros(len(names), dtype=bool)
    reference = None
    for run, name in enumerate(names):
        try:
            output = aims_stdout.AimsStdout(root / name / "aims.out")
            ok[run] = bool(
                output.is_finished_ok
                and output.converged
                and output.free_energy is not None
            )
        except Exception:  # noqa: BLE001  A missing or unreadable output.
            continue
        if not ok[run]:
            continue
        energies[run, 0] = output.free_energy
        for column, component in enumerate(components, 1):
            value = component.read(output)
            if value is not None:
                energies[run, column] = value
        times[run] = output.cpu_time or 0.0, output.total_time or 0.0
        if run == 0:
            reference = output
    if not ok.all():
        raise errors.FDError(
            f"not finished or not converged: {', '.join(names[~ok])}"
        )
    return energies, times, reference


def decomposition(
    energies: np.ndarray,
    components: tuple[EnergyComponent, ...] = ENERGY_COMPONENTS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Columns to difference and labels of the contributions (rest last)."""
    use = np.concatenate([[True], np.isfinite(energies[:, 1:]).all(axis=0)])
    labels = tuple(
        component.label
        for component, used in zip(components, use[1:], strict=True)
        if used
    )
    return energies[:, use], (*labels, REST_LABEL) if labels else ()


def with_rest(values: np.ndarray) -> np.ndarray:
    """Appends rest = total - components to a derivative.

    Taken after the differences, so that the contributions add up to the
    total exactly.
    """
    if len(values) == 1:
        return values
    rows = np.empty((len(values) + 1, *values.shape[1:]))
    rows[:-1] = values
    rows[-1] = values[0] - values[1:].sum(axis=0)
    return rows


def closure(values: np.ndarray) -> float:
    """Largest |sum of the contributions - total|; 0 without contributions."""
    if len(values) == 1:
        return 0.0
    return float(np.abs(values[1:].sum(axis=0) - values[0]).max())

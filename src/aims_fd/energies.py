from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyfhiaims.outputs.stdout import AimsStdout

from .errors import FDError


@dataclass(frozen=True)
class EnergyComponent:
    """An additive part of the free energy; read returns it in eV, or None if a run lacks it."""
    label: str
    read: Callable[[AimsStdout], float | None]


def pyfhiaims_value(key: str) -> Callable[[AimsStdout], float | None]:
    """Reader of a value that pyfhiaims parses into the last ionic step."""

    def read(output: AimsStdout) -> float | None:
        return output.results["ionic_steps"][-1].get(key)

    return read


ENERGY_COMPONENTS = (
    EnergyComponent("vdW energy correction", pyfhiaims_value("vdw_correction")),
)
REST_LABEL = "rest of the electronic free energy"


def collect(root: Path, names: np.ndarray,
            components: tuple[EnergyComponent, ...] = ENERGY_COMPONENTS):
    """Energies (n_runs x (1 + n_components), eV), times (n_runs x 2: cpu, wall) and the
    reference output. Column 0 is the free energy, the others the components (NaN where a run
    does not print one). Raises FDError if a run is unfinished or not converged.
    """
    energies = np.full((len(names), 1 + len(components)), np.nan)
    times = np.zeros((len(names), 2))
    ok = np.zeros(len(names), dtype=bool)
    reference = None
    for run, name in enumerate(names):
        try:
            output = AimsStdout(root / name / "aims.out")
            ok[run] = bool(output.is_finished_ok and output.converged
                           and output.free_energy is not None)
        except Exception:  # noqa: BLE001  (missing or unreadable output)
            continue
        if not ok[run]:
            continue
        energies[run, 0] = output.free_energy
        for column, component in enumerate(components, 1):
            value = component.read(output)
            if value is not None:
                energies[run, column] = value
        times[run] = output.cpu_time or 0.0, output.total_time or 0.0
        reference = output if run == 0 else reference
    if not ok.all():
        raise FDError(f"not finished or not converged: {', '.join(names[~ok])}")
    return energies, times, reference


def decomposition(energies: np.ndarray,
                  components: tuple[EnergyComponent, ...] = ENERGY_COMPONENTS):
    """The columns to difference (the free energy and every component that all runs print)
    and the labels of the contributions, the rest of the free energy last; no labels without
    components.
    """
    use = np.concatenate([[True], np.isfinite(energies[:, 1:]).all(axis=0)])
    labels = tuple(component.label for component, used in zip(components, use[1:], strict=True)
                   if used)
    return energies[:, use], (*labels, REST_LABEL) if labels else ()


def with_rest(values: np.ndarray) -> np.ndarray:
    """Append rest = total - components to (total, components...) of a derivative. Taken after
    the differences, so that the contributions add up to the total exactly."""
    if len(values) == 1:
        return values
    rows = np.empty((len(values) + 1, *values.shape[1:]))
    rows[:-1] = values
    rows[-1] = values[0] - values[1:].sum(axis=0)
    return rows


def closure(values: np.ndarray) -> float:
    """Largest |sum of the contributions - total|; 0 without contributions."""
    return float(np.abs(values[1:].sum(axis=0) - values[0]).max()) if len(values) > 1 else 0.0

import numpy as np

from .deformations import VOIGT


def fd_forces(energies: np.ndarray, n_atoms: int, delta: float) -> np.ndarray:
    """Forces (n_columns x n_atoms x 3, eV/Å) from the energies of the force runs
    (6 n_atoms x n_columns, in plan order)."""
    pairs = energies.reshape(n_atoms, 3, 2, -1)
    return np.moveaxis(-(pairs[:, :, 0] - pairs[:, :, 1]) / (2 * delta), -1, 0)


def fd_stress(energies: np.ndarray, strain_step: float, volume: float) -> np.ndarray:
    """Stress (n_columns x 3 x 3, eV/Å³) from the energies of the stress runs (12 x n_columns,
    in plan order); volume is that of the undistorted cell."""
    pairs = energies.reshape(6, 2, -1)
    voigt = (pairs[:, 0] - pairs[:, 1]) / (2 * strain_step * volume)
    stress = np.zeros((voigt.shape[1], 3, 3))
    stress[:, VOIGT[:, 0], VOIGT[:, 1]] = voigt.T
    stress[:, VOIGT[:, 1], VOIGT[:, 0]] = voigt.T
    return stress

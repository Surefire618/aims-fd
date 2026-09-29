"""Central finite differences of the run energies."""

import numpy as np

from aims_fd import deformations


def fd_forces(energies: np.ndarray, n_atoms: int, delta: float) -> np.ndarray:
    """Computes forces from the energies of the force runs.

    Args:
        energies: Energies of the force runs in plan order
            (6 n_atoms x n_columns, eV).
        n_atoms: Number of atoms.
        delta: Atom displacement in Angstrom.

    Returns:
        Forces of every column (n_columns x n_atoms x 3, eV/Angstrom).
    """
    pairs = energies.reshape(n_atoms, 3, 2, -1)
    forces = -(pairs[:, :, 0] - pairs[:, :, 1]) / (2 * delta)
    return np.moveaxis(forces, -1, 0)


def fd_stress(
    energies: np.ndarray, strain_step: float, volume: float
) -> np.ndarray:
    """Computes the stress from the energies of the stress runs.

    Args:
        energies: Energies of the stress runs in plan order
            (12 x n_columns, eV).
        strain_step: Strain step h.
        volume: Volume of the undistorted cell in Angstrom^3.

    Returns:
        Stress of every column (n_columns x 3 x 3, eV/Angstrom^3).
    """
    pairs = energies.reshape(6, 2, -1)
    voigt = (pairs[:, 0] - pairs[:, 1]) / (2 * strain_step * volume)
    rows, columns = deformations.VOIGT[:, 0], deformations.VOIGT[:, 1]
    stress = np.zeros((voigt.shape[1], 3, 3))
    stress[:, rows, columns] = voigt.T
    stress[:, columns, rows] = voigt.T
    return stress

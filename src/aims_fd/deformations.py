import copy

import numpy as np
from pyfhiaims.geometry import AimsGeometry

AXES = ("x", "y", "z")
SIGNS = np.array([1.0, -1.0])
SIGN_NAMES = ("plus", "minus")
COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")
VOIGT = np.array([[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]])


def plan(n_atoms: int, forces: bool, stress: bool, delta: float, strain_step: float):
    """Names, displacements u (n_runs x n_atoms x 3) and deformations D (n_runs x 3 x 3).

    Run 0 is the reference, followed by the force runs in the order (atom, axis, sign) and the
    stress runs in the order (component, sign), sign 0 being plus. A run's geometry is D (R + u).
    fd_forces and fd_stress rely on this order.
    """
    n_runs = 1 + 6 * n_atoms * forces + 12 * stress
    names = np.empty(n_runs, dtype=object)
    displacements = np.zeros((n_runs, n_atoms, 3))
    deformations = np.tile(np.eye(3), (n_runs, 1, 1))
    names[0] = "reference"
    run = 1
    for atom in range(n_atoms if forces else 0):
        for axis in range(3):
            for sign in range(2):
                names[run] = f"force_a{atom + 1:04d}_{AXES[axis]}_{SIGN_NAMES[sign]}"
                displacements[run, atom, axis] = SIGNS[sign] * delta
                run += 1
    for component in range(6 if stress else 0):
        i, j = VOIGT[component]
        for sign in range(2):
            names[run] = f"stress_{COMPONENTS[component]}_{SIGN_NAMES[sign]}"
            strain = np.zeros((3, 3))  # so that the diagonal of D is 1 ± h, rounded once
            strain[i, j] += SIGNS[sign] * strain_step / 2
            strain[j, i] += SIGNS[sign] * strain_step / 2
            deformations[run] += strain
            run += 1
    return names, displacements, deformations


def run_ranges(n_atoms: int, forces: bool, stress: bool) -> tuple[slice, slice]:
    n_force_runs = 6 * n_atoms * forces
    return slice(1, 1 + n_force_runs), slice(1 + n_force_runs, 1 + n_force_runs + 12 * stress)


def run_geometry(geometry: AimsGeometry, displacements: np.ndarray,
                 deformation: np.ndarray) -> AimsGeometry:
    new = copy.deepcopy(geometry)
    if new.lattice_vectors is not None:
        new.lattice_vectors = np.asarray(new.lattice_vectors, float) @ deformation.T
    for index, atom in enumerate(new.atoms):
        atom.position = deformation @ (np.asarray(atom.position, float) + displacements[index])
        # pyfhiaims writes atom_frac from cached fractional coordinates; a strain leaves them
        # unchanged, a displacement does not
        if new.lattice_vectors is not None and displacements[index].any():
            atom.set_fractional(np.asarray(new.lattice_vectors, float))
    return new

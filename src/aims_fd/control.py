"""Reading and editing control.in."""

# Commented out in the displaced and strained runs, which only need the energy.
REMOVED = (
    "compute_forces",
    "compute_analytical_stress",
    "compute_numerical_stress",
    "delta_numerical_stress",
    "sc_accuracy_forces",
    "sc_accuracy_stress",
)
# The reference keeps its analytical forces and stress for comparison; the
# built-in numerical stress would repeat the finite differences.
REMOVED_IN_REFERENCE = ("compute_numerical_stress", "delta_numerical_stress")
# Calculations that would be repeated in every run, or that change the
# geometry or the energy (the SCF free energy) that aims-fd differentiates.
# aims-fd stops if control.in has them.
UNSUPPORTED = (
    "relax_geometry",
    "relax_unit_cell",
    "path_integral",
    "use_pimd_wrapper",
    "communicate_pimd_wrapper",
    "plumed",
    "plumed_new",
    "calculate_atom_bsse",
    "phonon",
    "vibrations",
    "total_energy_method",
    "qpe_calc",
    "sc_self_energy",
    "neutral_excitation",
    "excited_mode",
    "excited_states",
    "TDDFT_run",
    "rrs_pbc",
    "output",
    "DFPT",
    "magnetic_response",
    "compute_dielectric",
    "compute_absorption",
    "compute_kubo_greenwood",
    "compute_greenwood_dc_transport",
    "compute_esp_charges",
    "compute_dipolematrix",
    "compute_dipolematrix_k_k",
    "compute_momentummatrix",
    "compute_heat_flux",
    "calculate_friction",
    "calculate_perturbative_soc",
    "calculate_mae",
    "calculate_plasma_frequency",
    "calculate_superconductivity",
    "transport",
    "evaluate_work_function",
)
UNSUPPORTED_PREFIXES = ("MD_", "PIMD_", "RT_TDDFT_")


def _tokens(line: str) -> list[str]:
    return line.split("#", 1)[0].split()


def last_value(control: str, keyword: str) -> list[str] | None:
    """Arguments of the last line of a keyword (the one FHI-aims uses)."""
    value = None
    for line in control.splitlines():
        tokens = _tokens(line)
        if tokens and tokens[0] == keyword:
            value = tokens[1:]
    return value


def requests(control: str) -> tuple[bool, bool, float]:
    """Forces requested, stress requested and the strain step."""

    def is_true(keyword):
        value = last_value(control, keyword)
        # Fortran logicals: .true., T, t, ...
        return bool(value) and value[0].lstrip(".")[:1] in "tT"

    step = (last_value(control, "delta_numerical_stress") or ["1e-4"])[0]
    return (
        is_true("compute_forces"),
        is_true("compute_analytical_stress")
        or is_true("compute_numerical_stress"),
        float(step.replace("d", "e").replace("D", "e")),
    )


def comment_out(control: str, removed: tuple[str, ...]) -> str:
    lines = control.splitlines()
    for index, line in enumerate(lines):
        tokens = _tokens(line)
        if tokens and tokens[0] in removed:
            lines[index] = f"# aims-fd removed: {line.strip()}"
    return "\n".join(lines) + "\n"


def unsupported(control: str) -> list[str]:
    """Lines of control.in with an unsupported keyword, as 'line N: text'."""
    found = []
    for number, line in enumerate(control.splitlines(), 1):
        tokens = _tokens(line)
        if tokens and (
            tokens[0] in UNSUPPORTED
            or tokens[0].startswith(UNSUPPORTED_PREFIXES)
        ):
            found.append(f"line {number}: {line.strip()}")
    return found

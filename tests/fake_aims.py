"""Stand-in for FHI-aims in the tests.

Prints data/si2_reference_aims.out with the inputs of the current directory echoed and the free
energy of a model with known forces FORCES and stress SIGMA; its quadratic terms cancel in
central differences. FAKE_AIMS_VDW=1 adds a vdW energy with forces FORCES_VDW and stress
SIGMA_VDW, included exactly in the free energy and printed with 8 decimals as FHI-aims does.
FAKE_AIMS_FAIL=<run> truncates the output of that run; FAKE_AIMS_LOG=<file> records every call.
"""

import gzip
import os
import re
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).parent / "data" / "si2_reference_aims.out.gz"
LATTICE = np.array([[0.0, 2.7155, 2.7155], [2.7155, 0.0, 2.7155], [2.7155, 2.7155, 0.0]])
POSITIONS = np.array([[0.0, 0.0, 0.0], [1.424186383883, 1.397611830330, 1.377680915165]])
SIGMA = np.array([[-0.010, 0.002, -0.003], [0.002, 0.004, 0.001], [-0.003, 0.001, 0.02]])
FORCES = np.array([[0.30, -0.20, 0.10], [-0.30, 0.20, -0.10]])
E0 = -15802.6
SIGMA_VDW = np.array([[0.002, 0.0, 0.001], [0.0, 0.003, 0.0], [0.001, 0.0, 0.004]])
FORCES_VDW = np.array([[0.012, 0.021, -0.033], [-0.012, -0.021, 0.033]])
E0_VDW = -0.46857337


def energy(geometry: str, sigma=SIGMA, forces=FORCES, e0=E0) -> float:
    lattice, positions = [], []
    for line in geometry.splitlines():
        tokens = line.split("#", 1)[0].split()
        if tokens[:1] == ["lattice_vector"]:
            lattice.append([float(v) for v in tokens[1:4]])
        elif tokens[:1] in (["atom"], ["atom_frac"]):
            positions.append(([float(v) for v in tokens[1:4]], tokens[0] == "atom_frac"))
    lattice = np.array(lattice)
    r = np.array([np.array(p) @ lattice if frac else p for p, frac in positions])
    deformation = np.linalg.solve(LATTICE, lattice).T
    strain = 0.5 * (deformation + deformation.T) - np.eye(3)
    u = r - POSITIONS @ deformation.T
    volume = abs(np.linalg.det(LATTICE))
    return (e0 + volume * np.sum(sigma * strain) + 0.3 * volume * np.sum(strain**2)
            - np.sum(forces * u) + 6.5 * np.sum(u**2))


def main():
    here = Path.cwd()
    if os.environ.get("FAKE_AIMS_LOG"):
        with open(os.environ["FAKE_AIMS_LOG"], "a") as log:
            log.write(here.name + "\n")
    with gzip.open(TEMPLATE, "rt") as handle:
        text = handle.read()
    for name in ("control.in", "geometry.in"):
        echo = "".join(f"  {line}\n" for line in (here / name).read_text().splitlines())
        text = re.sub(rf"(Parsing {re.escape(name)} .*?-{{71}}\n).*?(  -{{71}}\n  Completed)",
                      lambda m, e=echo: m.group(1) + e + m.group(2), text, count=1, flags=re.S)
    geometry = (here / "geometry.in").read_text()
    value = energy(geometry)
    if os.environ.get("FAKE_AIMS_VDW"):
        vdw = energy(geometry, SIGMA_VDW, FORCES_VDW, E0_VDW)
        value += vdw
        line = f"  | vdW energy correction         :   {vdw / 27.2113845:.8f} Ha   {vdw:.8f} eV\n"
        text = text.replace("  Self-consistency cycle converged.\n",
                            "  Self-consistency cycle converged.\n" + line, 1)
    text = re.sub(r"(\| (?:Electronic free energy|Total energy (?:un)?corrected)\s*:\s*)\S+( eV)",
                  lambda m: f"{m.group(1)}{value:.15E}{m.group(2)}", text)
    if os.environ.get("FAKE_AIMS_FAIL") == here.name:
        text = text[: len(text) // 2]
    print(text, end="")


if __name__ == "__main__":
    main()

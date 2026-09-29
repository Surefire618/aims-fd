# aims-fd

Finite-difference forces and stress for FHI-aims. It is a benchmark tool for
checking analytical forces and stress against the derivatives of the energy. The
number of FHI-aims runs grows with the number of atoms (see [Cost](#cost)), so
it is meant for small cells, not for production calculations.

aims-fd takes the `control.in` and `geometry.in` you would give FHI-aims. It
runs one FHI-aims calculation per displaced atom or strained cell. From their
energies it computes the forces and the stress, and it writes the result as an
`aims.out` that [pyfhiaims](https://gitlab.com/FHI-aims-club/pyfhiaims) and ASE
can read.

## Installation

```bash
pip install git+https://github.com/Surefire618/aims-fd.git
```

or, from a clone, `pip install .`. This needs Python ≥ 3.10, numpy and pyfhiaims
1.2.

## Basic usage

Use aims-fd as you would use `aims.x`. First, tell it once how you run FHI-aims,
in `~/.aimsfdrc`:

```ini
[machine]
aims_command = mpirun -np 8 /path/to/aims.x
```

Then ask for forces and/or stress in `control.in` with the usual keywords:

| control.in | aims-fd computes |
|---|---|
| `compute_forces .true.` | forces on all atoms |
| `compute_analytical_stress .true.` | the full stress tensor |
| `compute_numerical_stress .true.` | the full stress tensor |

Run it in the calculation directory:

```bash
cd my_calculation          # control.in + geometry.in
aims-fd > aims.out
```

`aims.out` is a regular FHI-aims output with the finite-difference forces and
stress:

```python
from pyfhiaims.outputs.stdout import AimsStdout

out = AimsStdout("aims.out")
out.forces, out.stress, out.free_energy
```

The individual runs are in `fd/`. `fd/reference/aims.out` is your calculation as
`aims.x` would have run it, including the analytical forces and stress to
compare with. Calling aims-fd again continues where it stopped.

### On a cluster

A Slurm job script:

```bash
#!/bin/bash
#SBATCH --job-name=aims-fd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --time=04:00:00
#SBATCH --partition=<partition>

module load impi intel                  # whatever your aims.x needs
source /path/to/venv/bin/activate       # the Python environment with aims-fd
ulimit -s unlimited
export OMP_NUM_THREADS=1

aims-fd > aims.out
```

Put this `.aimsfdrc` next to `control.in`:

```ini
[machine]
aims_command = srun /path/to/aims.x
```

Every FHI-aims run then uses the whole allocation, one run after another. If the
job ends before all runs have finished, submit it again; finished runs are kept.
For many small runs, see [Job farms](#job-farms).

## Advanced usage

### Options

| option | default | |
|---|---|---|
| `--forces` | what `control.in` asks for | compute the forces |
| `--stress` | what `control.in` asks for | compute the stress |
| `--delta` | 0.002 Å | atom displacement |
| `--strain-step` | `delta_numerical_stress`, else 1e-4 | strain step h |
| `--dir` | `fd` | directory of the runs |
| `--aims-command` | | overrides `.aimsfdrc` |
| `--dry` | | only write the runs (see [Job farms](#job-farms)) |
| `--version` | | versions of aims-fd and pyfhiaims |

`--forces` alone computes only the forces, `--stress` alone only the stress,
both flags both. Runs with `--forces` and with `--stress` can share one `fd/`;
the reference run is reused. The inputs and steps are recorded in
`fd/aims-fd.settings`. If they change, aims-fd refuses to reuse the directory;
delete it or pass another `--dir`.

### The FHI-aims command

aims-fd reads `aims_command` from `~/.aimsfdrc`, then from `./.aimsfdrc`; the
later file wins, and `--aims-command` overrides both. The command is run with
`bash -c` in every run directory, and its standard output becomes that run's
`aims.out`. Indented continuation lines form one script:

```ini
[machine]
aims_command =
    source /opt/intel/oneapi/setvars.sh
    ulimit -s unlimited
    mpirun -np 8 /path/to/aims.x
```

### Job farms

Small cells need many short runs that do not scale to a whole node. With
`--dry`, aims-fd only writes the run directories, so that a job farm can run
them side by side:

```bash
aims-fd --dry              # write fd/*/, report how many runs are not finished
# run FHI-aims in every fd/*/ directory, output aims.out
aims-fd > aims.out         # all runs finished: only evaluates
```

aims-fd needs the FHI-aims command only for runs that are not finished. If one
is still missing, the last step runs it where you call aims-fd. There is no
locking, so never start two aims-fd processes (without `--dry`) on the same
`fd/`.

### Energy decomposition

If the runs print a vdW energy (`| vdW energy correction`), `aims.out` also
splits the forces and the stress into contributions, after the totals:
- the finite differences of the vdW energy;
- the rest of the free energy, defined as total minus vdW.

The contributions therefore add up to the totals exactly. FHI-aims prints the
vdW energy with 8 decimals (1e-8 eV), which limits the vdW contribution to about
1e-8 eV / 2δ.

More components can be added in `ENERGY_COMPONENTS` in `aims_fd/energies.py`.
Append an `EnergyComponent(label, read)`, where `read(AimsStdout)` returns the
component in eV, or `None` if a run does not print it. A component is only split
out if every run prints it. `pyfhiaims_value(key)` reads a value that pyfhiaims
parses into the last ionic step.

## Algorithm

aims-fd computes the forces and the stress as central finite differences of the
electronic free energy E (`| Electronic free energy`). Each energy is one
FHI-aims run:

  F_iα = −[E(R + δ e_iα) − E(R − δ e_iα)] / 2δ

  σ_ij = [E(+h) − E(−h)] / (2 h V₀)

- **Forces.** `fd/force_a0003_y_plus/` moves atom 3 by +δ along y and leaves all
  other atoms in place. There is one such run for every atom, axis and sign.
  Atoms are numbered as in `geometry.in`.
- **Stress.** `fd/stress_xz_plus/` and `fd/stress_xz_minus/`, and likewise for
  the other components, apply the symmetric strain ε_xz = ε_zx = ±h/2 (ε_xx = ±h
  for the diagonal components) to the lattice vectors and to all atomic
  positions: r → (1 + ε) r. V₀ is the volume of the undistorted cell.
- **Reference.** `fd/reference/` is your calculation. Its `control.in` is yours,
  except that `compute_numerical_stress` and `delta_numerical_stress` are
  commented out. It provides the energy given in `aims.out` and the analytical
  forces and stress to compare with. The differences do not use it.

In the displaced and strained runs the lines `compute_forces`,
`compute_analytical_stress`, `compute_numerical_stress`,
`delta_numerical_stress`, `sc_accuracy_forces` and `sc_accuracy_stress` are
commented out, so these runs compute only the energy. Nothing is ever added to
`control.in`; the numerical settings, including the SCF convergence criteria,
are yours.

The forces are raw forces. FHI-aims by default prints them cleaned, with the net
force removed; set `final_forces_cleaned .false.` in `control.in` to compare
like with like.

**Accuracy.** A central difference has two errors:
- the truncation error (δ²/6) ∂³E/∂x³, which grows with the step;
- the noise σ_E/(√2 δ) for independent energy errors σ_E, which shrinks with the
  step. σ_E is set by the SCF convergence and the numerical settings in
  `control.in`. For example, σ_E = 1e-8 eV gives 0.004 meV/Å at δ = 0.002 Å.

For the stress, replace δ by h and divide by V₀. `--delta` and `--strain-step`
change the steps.

## Cost

Every energy is a full SCF of the whole cell:

| quantity | runs |
|---|---|
| forces on N atoms | 6N |
| stress | 12 |
| reference | 1 |

Forces and stress together need 6N + 13 runs: 25 for 2 atoms, 61 for 8, 397 for
64 and 1309 for 216. The displaced and strained runs compute no forces or
stress, so each costs less than the reference run. Even so, the total is roughly
6N + 12 times the cost of the calculation itself.

This is close to the minimum for energies alone:
- each run gives one number;
- the forces on N atoms have 3N − 3 independent components, because a rigid
  translation does not change the energy;
- an error of order δ² needs two energies per component.

So forces on all atoms at this accuracy need at least 6N − 6 runs. Cheaper
schemes either have an error of order δ (one-sided differences) or use the
analytical forces of the displaced runs, which a reference for those forces must
not do.

The runs are independent of each other. By default aims-fd runs them one after
another, each on the whole allocation; with `--dry` they can run side by side
(see [Job farms](#job-farms)).

## Calculations aims-fd refuses

aims-fd repeats your `control.in` in every run and differentiates only the SCF
electronic free energy. It therefore stops, names the lines, and writes nothing
if `control.in` asks for any of the following:
- another geometry: relaxation, MD (any `MD_` or `PIMD_` keyword), path
  integrals, i-PI or PLUMED, BSSE, phonons, vibrations;
- an energy other than the SCF free energy, or excited states:
  `total_energy_method`, GW (`qpe_calc`, `sc_self_energy`),
  `neutral_excitation`, TDDFT (including any `RT_TDDFT_` keyword),
  `excited_mode`, `excited_states`, `rrs_pbc`;
- property calculations: any `output` line, DFPT, magnetic response, dielectric
  and optical properties, dipole and momentum matrices, transport, friction, ESP
  charges, perturbative SOC, magnetic anisotropy, plasma frequency,
  superconductivity, work function, heat flux.

The full list is `UNSUPPORTED` in `aims_fd/control.py`.

## Output details

Besides the forces and the stress, `aims.out` contains:
- the echoed inputs;
- the header lines of the reference run;
- the energy of the reference run;
- the total time of all runs;
- the versions of aims-fd, pyfhiaims and FHI-aims.

ASE ≥ 3.23 reads it too (`ase.io.read("aims.out", format="aims-output")`). ASE
3.22 reads energy and forces but not the stress.

If a run did not finish or its SCF did not converge, aims-fd names it and writes
no `aims.out`. Unfinished runs are repeated on the next call. A run that
finished without a converged SCF is not repeated; delete its `aims.out` to run
it again.

## Limitations

- `empty` and `pseudocore` sites are not supported.
- Only symmetric strains are used. The off-diagonal stress therefore differs
  from FHI-aims' built-in `compute_numerical_stress`, which uses simple shears.

## Tests

```bash
pip install '.[tests]'
pytest
```

The tests use `tests/fake_aims.py` in place of FHI-aims.

## License

MIT, see [LICENSE](LICENSE).

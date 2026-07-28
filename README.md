# USCT One-Frequency Foundation

For a concise, self-contained handoff covering the project status, scientific
direction, current capabilities, missing pieces, professor-code comparison, and
next steps, start with [`AGENTS.md`](AGENTS.md).

This project is a small, inspectable forward simulator for an idealized
two-dimensional ultrasound computed-tomography (USCT) experiment. It accepts one
sound-speed field, one temporal frequency, and one or more known complex pressure
patterns on a circular boundary. It solves the scalar Helmholtz equation and
returns the complex outward boundary flux:

```text
known boundary pressure g  ->  complex interior pressure u  ->  du/dn
```

The result is an idealized **Dirichlet-to-Neumann (DtN) response**, stored as
`D = I + iQ`; neither phase nor quadrature is discarded. The time convention is
always `p(x,t) = Re{u(x) exp(-i omega t)}`.

This is USCT, not MRI. MRI observes nuclear magnetic resonance in a strong
magnetic field. This simulator models acoustic-wave transmission, with the
longer-term research motivation of obtaining useful soft-tissue information using
less expensive hardware.

## Installation on macOS

Install Python 3.14, then create an isolated environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Runtime dependencies are only NumPy, SciPy, scikit-fem, and Matplotlib. pytest and
Ruff are development dependencies. PETSc, MPI, Gmsh, compilers, GPU SDKs, and
machine-learning frameworks are not required.

## Command line

Explain the physical and array contracts:

```bash
python -m usct explain
```

Run the analytic disk, residual, boundary-enforcement, linearity, sign, and
three-mesh convergence checks:

```bash
python -m usct verify
```

Run the inexpensive lossy-water demonstration. Configuration is validated,
including its actual points per wavelength, before operator assembly:

```bash
python -m usct simulate configs/water_disk.toml --output runs/water
```

Plot the saved arrays without rerunning the simulation:

```bash
python -m usct plot runs/water
python -m usct plot runs/water --pattern cos:1
```

The simulation writes `result.npz` for numeric arrays and `metadata.json` for
readable experiment information. The plot command writes `overview.png`.

For a friendly but technically precise explanation of the proposed equipment,
data, inverse problem, adjoint method, frequency stepping, and cycle skipping,
read
[`docs/problem_definition.md`](docs/problem_definition.md).

## What experiment is modeled?

The implemented experiment prescribes complex acoustic pressure continuously on
the entire disk boundary and observes its outward normal derivative there:

```text
-Laplacian(u) - omega^2/c(x)^2 (1 + i eta) u = 0
u = g on boundary(Omega)
measurement = du/dn on boundary(Omega)
```

A source/receiver apparatus may instead emit a source or normal velocity at
discrete transducers and measure pressure elsewhere. That is a different boundary
condition and measurement operator. This project deliberately does not claim that
its DtN observable is already the calibrated hardware observable.

## What is verified?

The current foundation verifies:

- circular P1 triangular mesh geometry and refinement;
- sound-speed validation and explicit complex Helmholtz assembly;
- complex-symmetric operator structure and explicit loss behavior;
- exact Dirichlet elimination and one SciPy sparse LU for all forcing columns;
- machine-precision boundary enforcement and small interior residuals;
- weak boundary-mass projection of the outward Neumann trace;
- complex linearity and `exp(i m theta) = cos(m theta) + i sin(m theta)`;
- homogeneous-disk Bessel fields and fluxes for modes 0, 1, and 2;
- decreasing field and flux error across three successive refinements;
- phase-preserving NPZ/JSON round trips and saved-result plotting.

The central default test uses `kR = 3`, away from the relevant Bessel zeros.
The declared acceptance limits are 0.02 relative field error and 0.10 relative
flux error, measured with finite-element mass matrices.

This work does **not** verify a transducer model, density variation, three-dimensional
propagation, a realistic attenuation/dispersion law, calibration error, inversion,
optimization, adjoints, frequency stepping, regularization, or scalable/GPU solvers.

## Unconfirmed hardware questions

Before interpreting the DtN response as measured data, the research team must
confirm:

- whether a transmitter prescribes pressure, normal velocity, displacement, or
  only an electrical drive;
- which calibrated physical quantity a receiver returns;
- whether I/Q describes pressure, particle velocity, or a referenced transfer
  function;
- whether sources and receivers are discrete or effectively continuous patterns;
- whether density is known and constant;
- the appropriate attenuation and dispersion model;
- tank geometry, boundary material, operating frequencies, and bandwidth;
- how a water/reference scan is applied;
- whether source and receiver phases are individually calibrated; and
- whether the intended inverse datum is pressure, flux, or a DtN matrix.

The exact questions are maintained in
[`docs/questions_for_professor.md`](docs/questions_for_professor.md).

## Quality checks

```bash
pytest -q
ruff check .
```

The professor's original `alpha_expr` prototype has been fully absorbed into the
verified foundation (`usct.medium.PROFESSOR_BREAST_PHANTOM`, expressible as
`configs/breast_phantom.toml`) and the legacy prototype file itself removed; its
history remains in git.

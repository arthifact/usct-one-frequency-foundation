# Architecture

`src/usct` is organized so the two forward **sims** are separated, their shared
**physics** lives in one place, and it is obvious where to add what:

```text
physics/     shared primitives both sims build on:
             domain, medium, forcing, operator, boundary
dtn/         SIM 1 -- Dirichlet-to-Neumann (pressure in -> flux out):
             solver, measurement, simulation, verification,
             inversion, attenuation, reconstruction
radiating/   SIM 2 -- radiating/impedance (velocity in -> pressure out):
             forward, inversion
io/          config, results, plotting
cli.py       command-line interface (explain / verify / simulate / plot)
```

Dependency direction is acyclic, and **the two sims never import each other** --
each depends only on `physics`:

```text
physics   <── dtn          (sim 1 builds on shared physics)
physics   <── radiating    (sim 2 builds on shared physics)
physics   <── io.config
dtn.simulation <── io.results, io.plotting
cli ──> io, dtn
```

Numerical modules do not import the CLI, plotting, result I/O, or research
directory. Importing `usct` creates no mesh, matrix, files, or figures.

## The two sims have different roles

- **`radiating/` is the instrument-facing forward** — the physically faithful,
  open-medium model (no resonances; velocity-in / pressure-out, matching real
  transducers). Build the real reconstruction path on this one.
- **`dtn/` is the verified reference** — an idealized closed-cavity experiment
  kept as the analytically-verifiable gold standard (cleanest exact Bessel
  solution; the full `python -m usct verify` trust battery) and as a differential
  cross-check. Not the physical instrument, but the oracle the numerics are proven
  against.

`dtn/` currently has more modules simply because it is the older, mature sim: it
carries the whole forward pipeline (`solver`, `measurement`, `simulation`,
`verification`) plus the first research layer (`inversion`, `attenuation`,
`reconstruction`). `radiating/` is leaner (`forward` + `inversion`) because its
full solve and its trivial boundary-pressure observable need no separate
`solver`/`measurement` modules, and its experiments currently drive reconstruction
inline. As the instrument path matures, radiating may grow its own driver /
attenuation variant (ideally by generalizing the DtN ones into shared code).

## Files

| File | Responsibility |
| --- | --- |
| `AGENTS.md` | Canonical short handoff for future humans and coding agents: current status, scientific direction, comparisons, decisions, open questions, and next steps. |
| `pyproject.toml` | Declares the Python 3.14 package, the four runtime dependencies, pytest/Ruff development tools, and quality settings. |
| `configs/analytic_disk.toml` | Dimensionless lossless `kR = 3` Bessel-verification experiment. |
| `configs/water_disk.toml` | Low-cost SI water-disk demonstration with explicitly configured loss. |
| `configs/tissue_disk.toml` | SI demonstration with one centered circular sound-speed inclusion in water; it is not the professor's full phantom. |
| `src/usct/__init__.py` | Exports immutable configuration records and the one public simulation entry point; performs no work. |
| `src/usct/__main__.py` | Connects `python -m usct` to the CLI. |
| **`src/usct/physics/`** | **Shared primitives used by both sims.** |
| `src/usct/physics/domain.py` | Builds a truth-independent, uniformly refined scikit-fem disk and deterministic degree-of-freedom partitions. |
| `src/usct/physics/medium.py` | Constructs validated constant, circular-inclusion, or feature-phantom nodal sound-speed arrays. |
| `src/usct/physics/forcing.py` | Parses and evaluates only the declared constant and Fourier boundary patterns. |
| `src/usct/physics/operator.py` | Assembles the inspectable complex128 Helmholtz operator; also holds the shared FEM forms (stiffness, adjoint-gradient load) used by both sims' inversions. |
| `src/usct/physics/boundary.py` | Shared boundary tooling: the L2 boundary mass matrix and periodic cross-mesh flux resampling. |
| **`src/usct/dtn/`** | **Sim 1: Dirichlet-to-Neumann (pressure in, flux out).** |
| `src/usct/dtn/solver.py` | Applies explicit Dirichlet block elimination and one batched sparse LU, reporting residual and timing diagnostics. |
| `src/usct/dtn/measurement.py` | Recovers the outward Neumann flux by a boundary-mass projection and returns angle-sorted complex samples. |
| `src/usct/dtn/simulation.py` | Orchestrates exactly one boundary-driven experiment and records assumptions and dependency versions. |
| `src/usct/dtn/verification.py` | Implements Bessel, sign, residual, linearity, Fourier-identity, and three-mesh convergence checks. |
| `src/usct/dtn/inversion.py` | Complex-L2 flux-misfit objective and adjoint-state gradient (research). |
| `src/usct/dtn/attenuation.py` | Complex-medium forward and joint `(c, eta)` adjoint gradient (research). |
| `src/usct/dtn/reconstruction.py` | Bounded L-BFGS-B frequency-continuation FWI driver (research). |
| **`src/usct/radiating/`** | **Sim 2: radiating/impedance boundary (velocity in, pressure out).** |
| `src/usct/radiating/forward.py` | Open-boundary forward solve; boundary-pressure observable; analytic Bessel reference. |
| `src/usct/radiating/inversion.py` | Complex-L2 boundary-pressure objective and adjoint-state gradient (research). |
| **`src/usct/io/`** | **Configuration and result I/O.** |
| `src/usct/io/config.py` | Loads TOML into frozen records, validates values and forcing names, and enforces actual points per wavelength. |
| `src/usct/io/results.py` | Saves numeric arrays to NPZ, readable metadata to JSON, and loads with pickle disabled. |
| `src/usct/io/plotting.py` | Uses the actual triangular mesh to create a seven-view, noninteractive overview from saved data. |
| `src/usct/cli.py` | Implements `explain`, `verify`, `simulate`, and saved-result `plot` with `argparse`. |
| `tests/conftest.py` | Shares small deterministic numerical fixtures without global solver state. |
| `tests/test_config.py` | Tests immutable TOML records, invalid inputs, pattern validation, and wavelength diagnostics. |
| `tests/test_domain.py` | Tests geometry, degree-of-freedom partitioning, deterministic ordering, and refinement. |
| `tests/test_forcing.py` | Tests the exact pattern grammar, boundary angles, and complex Fourier values. |
| `tests/test_operator.py` | Tests shape, complex dtype, finite entries, symmetry, parameter effects, and explicit loss. |
| `tests/test_solver.py` | Tests boundary enforcement, residuals, batched/separate agreement, and one factorization per experiment. |
| `tests/test_analytic_disk.py` | Tests non-resonant Bessel field/flux accuracy and outward sign. |
| `tests/test_linearity.py` | Tests arbitrary complex superposition and exponential/cosine/sine identities. |
| `tests/test_convergence.py` | Requires substantial error reduction on three successive meshes. |
| `tests/test_results.py` | Tests phase-preserving persistence, no-pickle loading, metadata, and PNG output. |
| `docs/problem_definition.md` | Formally defines the proposed steady-state modal tomography system, inverse problem, cost hypothesis, and validation path in accessible language. |
| `docs/mathematics.md` | Derives the time convention, weak form, elimination, and weak flux recovery. |
| `docs/questions_for_professor.md` | Tracks unresolved physical source, receiver, tank, and calibration questions. |
| `research/README.md` | Separates the inverse-research layer from the trusted forward foundation. |

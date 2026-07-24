# Architecture

The foundation has a single dependency direction:

```text
config
  ├──> domain
  ├──> medium
  └──> forcing

domain + medium ──> operator
operator + forcing ──> solver
operator + solver ──> measurement
foundation pieces ──> simulation
simulation ──> results and plotting
CLI ──> simulation, verification, results, plotting
```

Numerical modules do not import the CLI, plotting, result I/O, legacy program, or
research directory. Importing `usct` creates no mesh, matrix, files, or figures.

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
| `src/usct/config.py` | Loads TOML into frozen records, validates values and forcing names, and enforces actual points per wavelength. |
| `src/usct/domain.py` | Builds a truth-independent, uniformly refined scikit-fem disk and deterministic degree-of-freedom partitions. |
| `src/usct/medium.py` | Constructs validated constant or circular-inclusion nodal sound-speed arrays. |
| `src/usct/operator.py` | Assembles inspectable complex128 stiffness, wave-mass, and Helmholtz matrices. |
| `src/usct/forcing.py` | Parses and evaluates only the declared constant and Fourier boundary patterns. |
| `src/usct/solver.py` | Applies explicit Dirichlet block elimination and one batched sparse LU, reporting residual and timing diagnostics. |
| `src/usct/measurement.py` | Recovers the outward Neumann trace by one boundary-mass factorization and returns angle-sorted complex samples. |
| `src/usct/simulation.py` | Orchestrates exactly one boundary-driven experiment and records assumptions and dependency versions. |
| `src/usct/results.py` | Saves numeric arrays to NPZ, readable metadata to JSON, and loads with pickle disabled. |
| `src/usct/plotting.py` | Uses the actual triangular mesh to create a seven-view, noninteractive overview from saved data. |
| `src/usct/verification.py` | Implements Bessel, sign, residual, linearity, Fourier-identity, and three-mesh convergence checks. |
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
| `research/README.md` | Separates future inverse-research hypotheses from the trusted forward foundation. |
| `legacy/README.md` | Defines historical code as non-importable evidence only. |

# USCT One-Frequency Foundation — Durable Project Context

Last updated: 2026-07-24

This is the short, canonical handoff for humans, coding agents, and future chat
sessions. Read it before changing the project. For deeper explanations, follow
the links in **Key files**. New user instructions and an explicitly supplied
authoritative specification take precedence over this summary.

## 1. Executive summary

The long-term goal is a potentially lower-cost acoustic tomography system that
produces useful quantitative soft-tissue information. The working hypothesis is
to:

1. surround a target with an acoustic transducer ring;
2. drive calibrated spatial boundary modes at one sinusoidal frequency at a time;
3. use coherent I/Q demodulation to retain complex amplitude and phase;
4. step from low to higher frequencies;
5. reconstruct the internal sound-speed field with an adjoint-based inverse method
   and an explicit structural prior.

The possible cost advantage would come from narrowband excitation, baseband I/Q
measurements, and moving some complexity from acquisition hardware into
computation. This remains a research hypothesis, not a proven hardware design or
novelty claim.

The current project implements and verifies only the trusted forward foundation:

```text
known sound speed c(x)
+ one frequency
+ one or more known boundary-pressure patterns
        ↓
complex Helmholtz solve
        ↓
ideal complex outward boundary-flux response D = I + iQ
```

There is no inversion in the current `src/usct` package.

## 2. Exact mathematical contract currently implemented

Time convention:

```text
p(x,t) = Re{u(x) exp(-i omega t)}
u = I + iQ
p(x,t) = I cos(omega t) + Q sin(omega t)
```

Material and loss model:

```text
omega = 2 pi f
m(x) = 1 / c(x)^2
k_squared(x) = omega^2 m(x) (1 + i eta)
```

Forward problem:

```text
-Laplacian(u) - k_squared(x) u = 0       in the disk
u = g_s                                  on the boundary
D_s = du_s/dn                            on the boundary
```

This is an idealized Dirichlet-to-Neumann (DtN) experiment:

```text
known boundary pressure -> measured outward flux
```

Supported boundary patterns:

```text
constant
cos:m
sin:m
exp:m
```

The complex response is primary. Magnitude and phase are derived views:

```text
I = real(D)
Q = imag(D)
amplitude = abs(D)
phase = angle(D)
```

## 3. What the meeting notes most likely mean

| Meeting phrase | Working interpretation |
| --- | --- |
| Steady-state / Helmholtz | Solve one harmonic frequency at a time |
| Whole system vibrates at one frequency | Same temporal frequency everywhere, different spatial amplitude and phase |
| I/Q demodulation | Synchronized cosine/sine detection of the complex response |
| DC amplitude | Low-pass demodulation produces baseband I and Q; do not discard Q |
| Boundary forcing pattern | Coherently drive angular Fourier modes around a ring |
| Frequency cycle/stepping | Invert low frequencies first, then add higher frequencies |
| Adjoint method | Efficient gradient for all sound-speed unknowns |
| Impose structure | Probably regularization or a prior; exact meaning is unconfirmed |
| Create data set | Generate controlled synthetic observations |
| Try to invert it | Recover `c(x)` by matching predicted and observed complex data |

The strongest clue to a potentially distinctive idea is the combination of
coherent boundary Fourier modes, narrowband I/Q acquisition, and
frequency-stepped reconstruction. Ring-array USCT, FWI, low-frequency
continuation, and frequency differencing already exist in the literature, so
the professor's exact new claim must be stated more narrowly and confirmed
directly.

Important correction: “compression” was a discarded interpretation added by
the note-taker, not something the professor is known to have said. Do not
attribute a data-compression hypothesis to the professor without new evidence.

## 4. Current project vs. professor prototype vs. real equipment

| Capability | Current project | `old_lowf_inv.py` | Real system |
| --- | --- | --- | --- |
| Steady-state Helmholtz | Verified | Present | Plausible |
| Complex I/Q phasor | Ideal representation | Ideal representation | Requires calibrated demodulation |
| Cosine/sine boundary modes | Verified | Present | Must prove the ring can synthesize them |
| Boundary pressure -> flux | Verified mathematically | Intended | Hardware observable unconfirmed |
| Multiple modes | One batched LU | Looped solves | Acquisition design unconfirmed |
| Frequency stepping | Not implemented | Attempted | Future experiment |
| Synthetic data | Saved forward results | Multi-frequency generation | Physical data later |
| Adjoint gradient | Not implemented | Attempted, not trusted | Future software |
| Inversion | Not implemented | Attempted with L-BFGS-B | Future milestone |
| Structural prior | Not implemented | Partial/unclear | Must be explicit |
| Noise/calibration/3D | Not modeled | Not modeled | Unavoidable |

Interpretation:

- The current project matches the professor's **idealized forward direction**.
- The legacy program is a rough prototype of the broader reconstruction loop.
- Neither codebase establishes that the ideal boundary condition and flux
  measurement match physical transducers.

## 5. Current verified status

Snapshot as of 2026-07-24:

- Python 3.14, NumPy, SciPy, scikit-fem, Matplotlib, pytest, and Ruff only.
- `pytest -q`: 33 tests pass.
- `ruff check .`: passes.
- `python -m usct verify`: all analytic, residual, sign, linearity, boundary,
  Fourier-identity, and convergence checks pass.
- Homogeneous analytic disk, refinement 5:
  - 2,113 degrees of freedom;
  - maximum relative field error: approximately `0.002405`;
  - maximum relative flux error: approximately `0.05833`;
  - acceptance limits: `0.02` and `0.10`.
- Errors decrease substantially across refinements 3, 4, and 5.
- One Helmholtz LU factorization serves every forcing column at a frequency.
- Complex arrays serialize as `complex128` with pickle disabled.

Demonstration runs:

- `runs/water`: homogeneous `1500 m/s` disk.
- `runs/tissue`: simple centered circular inclusion, `1540 m/s` inside a
  `1500 m/s` background.
- Both use 8,321 DOFs, 256 boundary samples, 5 patterns, and 25 kHz.
- The tissue run is **not** the professor's full gland/lesion/skin phantom.
- `overview.png` defaults to the first (`constant`) pattern. Dark-teal subtitles
  are plain-language reading aids, not additional simulated quantities.

## 6. Important design decisions already made

- Use a uniform truth-independent circular mesh. Do not refine around known
  lesions; that leaks the answer into an inverse problem.
- Use scikit-fem P1 triangles and inspectable `K`, `H`, and `A = K - H`.
- Use `complex128` throughout.
- Apply Dirichlet values through explicit interior/boundary block elimination.
- Factor the one-frequency interior matrix once and solve all patterns together.
- Recover flux weakly through a boundary-mass projection, not a noisy pointwise
  P1 gradient.
- Retain I and Q; never reduce saved data to amplitude only.
- Keep forward numerics independent of plotting, result I/O, CLI, legacy, and
  future research code.
- Do not import or extend `old_lowf_inv.py`; it is historical evidence only.
- Do not add inversion, optimization, adjoints, frequency schedules, GPU paths,
  PETSc, MPI, Gmsh, or new dependencies without an explicitly authorized
  milestone.

## 7. What is implemented and what is not

Implemented and trusted:

- configuration validation and points-per-wavelength guard;
- circular mesh and deterministic DOF partition;
- constant, one-circular-inclusion, and feature-phantom (disk/ellipse/ring)
  sound-speed models, all expressible as TOML configs (the professor's breast
  phantom is `configs/breast_phantom.toml`);
- complex Helmholtz assembly;
- named boundary Fourier patterns;
- batched sparse direct solve;
- weak Neumann trace;
- I/Q, amplitude, and phase views;
- NPZ/JSON persistence;
- saved-result plotting;
- analytic and automated verification.

Implemented as **research** (2026-07-27, authorized milestone; layered on top of
the verified foundation without modifying it — see `docs/inversion_hypothesis.md`):

- nodal sound-speed inversion parameterization;
- complex-L2 boundary-flux objective in the `L2(boundary)` inner product,
  with H1 (Tikhonov) regularization (`src/usct/inversion.py`);
- adjoint-state gradient, **verified** by central-difference and second-order
  Taylor tests (`tests/test_adjoint.py`) — the forward LU factorization is
  re-used for the adjoint solve;
- frequency-continuation L-BFGS-B reconstruction driver
  (`src/usct/reconstruction.py`);
- different-mesh synthetic observations with additive noise, and a first
  end-to-end recovery + cold-start (cycle-skipping) control
  (`research/fwi_experiment.py`).

Not implemented:

- measured-data import;
- discrete transmitter and receiver geometry;
- absorbing or experimentally calibrated boundaries;
- actual I/Q electronics;
- physical boundary-mode acquisition and identifiability study;
- density or attenuation inversion;
- kR scaling study and the direct-solver wall;
- high-contrast/high-kR frozen cycle-skipping regression;
- physical phantom or human data;
- 3D propagation.

## 8. The future inverse problem

For observed complex data `D_observed`, a conventional control problem would be

```text
minimize over c:

    J(c) =
      1/2 sum_{frequency, pattern}
          ||D_predicted(c) - D_observed||^2
      + alpha R(c),

subject to c_min <= c(x) <= c_max.
```

The forward simulator provides `D_predicted(c)`. A future adjoint calculates the
gradient efficiently. Frequency continuation uses the low-frequency result to
initialize the next frequency.

Cycle skipping is a failure mode of this optimization, not the overall goal. It
occurs when predicted and observed phases align to the wrong oscillation, leading
to a false local minimum.

## 9. Required next steps, in order

1. Ask the professor to state the one-sentence new claim.
2. Confirm what **impose structure** means.
3. Confirm the transmitter quantity, receiver quantity, geometry, calibration,
   frequency band, and whether modal patterns are physically synthesized.
4. If hardware is discrete source/pressure receiver, implement a separate
   forward experiment; do not silently change the verified DtN function.
5. Define arbitrary candidate `c(x)` values and generate observations on a
   different mesh to avoid an inverse crime.
6. Implement conventional complex-L2 FWI as the control.
7. Verify the adjoint with Taylor tests.
8. Freeze a reproducible cycle-skipping failure.
9. Test frequency stepping, phase/beat initialization, or the professor's
   specific innovation against that control.
10. Progress from independent synthetic data to water-tank and tissue-mimicking
    phantoms before making real-world claims.

## 10. Open questions that must not be silently assumed

- What acoustic quantity does a transmitter actually prescribe?
- What calibrated quantity does a receiver return?
- Does I/Q represent pressure, velocity, flux, or an electrical transfer function?
- Are sources discrete elements or simultaneous modal drives?
- What structural prior is intended?
- Which cost is expected to decrease: elements, digitizers, bandwidth,
  acquisitions, computation, or something else?
- What existing scanner/algorithm is the baseline?
- What body part, resolution, and clinical use are targeted?
- What experimental result would falsify the central hypothesis?

## 11. Key files

- `docs/problem_definition.md`: full friendly and technical research definition.
- `docs/mathematics.md`: derivation of the verified forward experiment.
- `docs/architecture.md`: module responsibilities and dependency direction.
- `docs/questions_for_professor.md`: detailed hardware questions.
- `README.md`: installation and user commands.
- `src/usct/simulation.py`: current one-frequency forward orchestration.
- `src/usct/operator.py`: inspectable Helmholtz operator.
- `src/usct/solver.py`: explicit batched Dirichlet solve.
- `src/usct/measurement.py`: weak outward-flux recovery.
- `src/usct/verification.py`: analytic and convergence checks.
- `research/README.md`: explicitly deferred research.
- `old_lowf_inv.py`: legacy evidence; never a dependency.

## 12. Commands to re-establish confidence

From the project virtual environment:

```bash
python -m usct explain
python -m usct verify
pytest -q
ruff check .
python -m usct simulate configs/water_disk.toml --output runs/water
python -m usct plot runs/water
python -m usct simulate configs/tissue_disk.toml --output runs/tissue
python -m usct plot runs/tissue
```

## 13. Rules for future sessions

- Read this file before acting.
- Preserve the verified forward foundation while adding higher-level research.
- Distinguish foundation, modeling assumptions, and research hypotheses.
- Do not claim that acoustic imaging is physically equivalent to MRI.
- Do not claim hardware viability from same-model synthetic reconstruction.
- Clearly label inference about the professor's idea until he confirms it.
- Update this file when a milestone changes what is implemented, verified,
  explicitly excluded, or still uncertain.

# Research layer — verified inversion on top of the trusted foundation

This directory holds the inverse-problem research, layered on the verified
one-frequency forward simulator **without ever modifying it**. `src/usct`
foundation code must never import from `research`, and experiments must not change
the meaning of the verified Dirichlet-to-Neumann simulator.

## Implemented and verified

The research modules live in `src/usct` (covered by tests) and are exercised by
the experiments here:

- **Adjoint-state FWI** (`usct.inversion`) — complex-L2 boundary-flux objective and
  adjoint gradient, verified by central finite differences and a Taylor test
  (`tests/test_adjoint.py`). Driver: `usct.reconstruction` (L-BFGS-B, frequency
  continuation). See `docs/inversion_hypothesis.md`.
- **Attenuation** (`usct.attenuation`) — spatially-varying loss field with a
  verified joint `(c, eta)` gradient. See `docs/attenuation.md`.
- **Radiating boundary** (`usct.radiating`) — the open/impedance forward, analytic-
  and adjoint-verified. See `docs/radiating_boundary.md`.

Experiments (each writes to `research/outputs/`, which is gitignored):

- `fwi_experiment.py` — end-to-end sound-speed recovery + cycle-skipping control.
- `attenuation_experiment.py` — resonance damping and attenuation imaging.
- `radiating_experiment.py` — resonances removed at the source; boundary-pressure
  reconstruction.
- `kr_scaling.py` — the direct-solver cost and pollution walls (`docs/kr_scaling.md`).
- `si_reconstruction.py` — realistic few-percent contrast in SI units
  (`docs/si_reconstruction.md`).
- `noise_tolerance.py` — reconstruction error vs measurement noise / SNR
  (`docs/noise_tolerance.md`).

## Still deferred

Phase-difference / beat-frequency initialization, IR-WRI, optimal-transport
objectives, learned priors, and scalable iterative solvers (GMRES, matrix-free,
custom Helmholtz preconditioners) for high kR / 3D — the last of these motivated by
`docs/kr_scaling.md`. If hardware uses discrete sources and pressure receivers,
that discrete-element model belongs in a separate forward beside the verified DtN
(the radiating boundary is the first such faithful forward).

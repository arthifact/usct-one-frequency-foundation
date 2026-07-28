# Research

The inverse-problem research, layered on the verified one-frequency forward
simulator **without ever modifying it**. `src/usct` foundation code must never
import from `research`.

## How this is organized

- **[`LOG.md`](LOG.md)** — the chronological story: what each step found and **why
  it led to the next**. Start here.
- **One numbered folder per step**, in order. Each holds a runnable `experiment.py`
  and an `outputs/` directory (gitignored) where its figures/data are written:

  | # | folder | topic | write-up |
  | --- | --- | --- | --- |
  | 1 | `01-adjoint-fwi/` | adjoint FWI, end-to-end recovery, cycle-skipping control | `docs/inversion_hypothesis.md` |
  | 2 | `02-kr-scaling/` | direct-solver cost and pollution walls | `docs/kr_scaling.md` |
  | 3 | `03-attenuation/` | attenuation as a physical field, joint `(c, eta)` adjoint | `docs/attenuation.md` |
  | 4 | `04-radiating-boundary/` | open/impedance boundary (resonances removed) | `docs/radiating_boundary.md` |
  | 5 | `05-si-reconstruction/` | realistic few-percent contrast in SI units | `docs/si_reconstruction.md` |
  | 6 | `06-noise-tolerance/` | reconstruction error vs measurement noise / SNR | `docs/noise_tolerance.md` |
  | 7 | `07-calibration-tolerance/` | reconstruction error vs systematic element gain/phase miscalibration | `docs/calibration_tolerance.md` |

The verified research *modules* those experiments call live in `src/usct`
(`inversion`, `reconstruction`, `attenuation`, `radiating`) and are covered by
tests. Run any step with, e.g., `python research/01-adjoint-fwi/experiment.py`.

## Still deferred

Calibration-tolerance (systematic gain/phase) studies, discrete finite-`N` element
drives, phase-difference / beat-frequency initialization, IR-WRI, optimal-transport
objectives, learned priors, and scalable iterative solvers (GMRES, matrix-free,
custom Helmholtz preconditioners) for high `kR` / 3D — the last motivated by
`docs/kr_scaling.md`.

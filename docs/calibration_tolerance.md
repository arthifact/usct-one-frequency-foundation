# Foundation fidelity: the calibration-tolerance curve

Companion to the noise-tolerance study. That study found random measurement noise
is *not* the binding constraint (usable to ~14 dB SNR), which raised the obvious
question: then what is? The prime suspect is **systematic** per-element
miscalibration -- a fixed complex gain (amplitude + phase) error on each element
that is consistent across every frequency and drive pattern, and therefore
**coherent** (it does not average out the way random noise does).

`research/07-calibration-tolerance/experiment.py`, on the instrument-facing
radiating sim, same phantom/continuation as `si_reconstruction` / `noise_tolerance`.

## Model

Each boundary element `j` gets a fixed complex gain `g_j = (1 + a_j) exp(i phi_j)`
with `a_j ~ N(0, level)` (amplitude) and `phi_j ~ N(0, level rad)` (phase), drawn
**once per device** (seed) and applied to the observed boundary pressure at every
stage. The inversion uses the nominal `g = 1` forward, so the miscalibration is an
unmodeled, frequency-coherent data error. Level is swept and reported in both
amplitude-% and phase-degrees.

## Result

| per-element error (amp / phase) | relative RMS (calibration) | relative RMS (random noise, step 06) |
| --- | --- | --- |
| 0 | 0.289 | 0.289 |
| 5% / 2.9° | 0.296 | 0.294 |
| 10% / 5.7° | 0.317 | 0.308 |
| 20% / 11.5° | 0.396 | 0.363 |
| 35% / 20° | **0.580 (breakdown)** | 0.483 |
| 50% / 29° | 0.823 | 0.626 |

Two readings:

1. **Calibration error is the harder error, as predicted.** At every level the
   calibration curve sits above the random-noise curve, and it breaks (relative
   RMS > 0.5) near **~30-35%** versus **~50%** for noise. The frequency-coherence
   is what makes the fixed gain more damaging than independent per-measurement
   noise of the same size.
2. **But the reconstruction is still fairly tolerant.** Recovery stays usable
   (relative RMS < ~0.4) up to about **20% amplitude / ~11° phase**, and near-clean
   at **5% / ~3°**. So the hardware calibration spec this implies is *achievable*:
   per-element amplitude matched to a few percent and phase to a few degrees gives
   essentially uncorrupted reconstruction.

## The important caveat (what this does and doesn't test)

This tests **spatially-random** per-element error (each element independent). That
is likely the *least* damaging kind of calibration error, because a spatially
incoherent gain is a high-spatial-frequency boundary perturbation that no smooth
interior `c(x)` can mimic, and it is further low-passed by the mesh resampling --
so it behaves partly like noise.

The genuinely dangerous case is **spatially-correlated** miscalibration -- a smooth
gain drift around the ring, a phase ramp, or a one-sided offset -- which *can*
alias into apparent interior structure and would be expected to break recovery at a
much lower level. That case is not tested here and is the natural follow-up. So the
honest conclusion is: *random* per-element calibration error is tolerable to a
loose, achievable spec; *structured* calibration error is the open risk.

## Where this leaves foundation fidelity

- Random noise and random per-element calibration are both non-binding at
  achievable hardware quality; **resolution (kR) remains the dominant limit**.
- Open: spatially-correlated calibration error; discrete finite-`N` element drives
  (element count vs resolution); 3D. And the kR wall (`docs/kr_scaling.md`) still
  gates clinical resolution.

# Foundation fidelity: the noise-tolerance curve

Replaces "1% noise worked once" with a measured breakdown curve — which *is* the
hardware SNR requirement. `research/06-noise-tolerance/experiment.py` sweeps the measurement-
noise fraction on the realistic-contrast SI radiating setup (same phantom and
continuation as `docs/si_reconstruction.md`), two independent noise draws per
level, and records how reconstruction error grows. Clean boundary data is computed
once and reused; only fresh noise is added per draw.

Noise is expressed as a fraction of the RMS boundary pressure, so
`SNR (dB) = 20 log10(1 / fraction)`.

## Result

| noise (% of signal RMS) | receiver SNR | relative RMS error | verdict |
| ---: | ---: | ---: | --- |
| 0 | ∞ | 0.289 | resolution-limited floor |
| 2 | 34 dB | 0.290 | unchanged |
| 5 | 26 dB | 0.294 | unchanged |
| 10 | 20 dB | 0.308 | excellent |
| 20 | 14 dB | 0.363 | clearly usable (all features visible) |
| 35 | 9 dB | 0.483 | degrading |
| **50** | **6 dB** | **0.626** | **breakdown (error > 0.5)** |
| 75 | 2.5 dB | 0.883 | mostly noise |
| 100 | 0 dB | 1.149 | worse than a homogeneous guess |

The min–max band over random seeds is within ±0.003 at every level, so this is a
deterministic curve, not a lucky draw.

## What it says

1. **In the practical regime the reconstruction is resolution-limited, not
   noise-limited.** Below ~10% noise the error is pinned at its noise-free floor
   (0.289) — that floor is set by kR (low-frequency diffraction limit), not by the
   receivers. Improving hardware SNR beyond ~20 dB buys nothing here; improving kR
   would.
2. **Graceful, predictable degradation.** From 10% to 50% the error rises smoothly
   and the features stay recognizable; there is no cliff until noise approaches the
   signal itself.
3. **A lenient hardware SNR requirement.** Usable reconstruction (error < ~0.4)
   holds down to **~14 dB SNR (20% noise)**; the knee is at **~6 dB (50% noise)**.
   Real ultrasound receive electronics comfortably exceed 14 dB, so **measurement
   noise is not the binding constraint** for this contrast/resolution regime.

## Honest scope

- This is *random* measurement noise. It says nothing about *systematic* errors —
  element gain/phase miscalibration, positioning error, or model mismatch — which
  couple coherently and are typically far more damaging than random noise. Those
  belong in a calibration-tolerance study (future).
- The floor (0.289) and the whole curve are tied to kR ≤ 16; at higher resolution
  the noise sensitivity would change.
- Noise is added to the synthetic forward data; real I/Q noise has its own
  (frequency-dependent, possibly correlated) structure.

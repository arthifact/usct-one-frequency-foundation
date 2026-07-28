# Foundation fidelity: realistic-contrast reconstruction in SI units

The honest translation test. Everything the earlier proofs did in dimensionless
units with comfortable 8-15% contrasts, redone in **real SI units** at the
**few-percent contrast real soft tissue actually shows** — on the physically
faithful radiating boundary. `research/si_reconstruction.py`; the dimensional
correctness it depends on is locked by a test (below).

## Setup (water-tank / breast scale)

- disk radius **0.10 m**, water bath **1500 m/s** (known, frozen at the boundary);
- gland **+1.7%**, tumour **+3.7%** (in glandular tissue, so +5.3% peak), cyst
  **−2.3%** — every contrast under the realistic few-percent tissue range;
- frequency continuation at **kR = 5, 10, 16** → **f = 11.9, 23.9, 38.2 kHz**
  (low-frequency regime, matching the project's kR ≲ 50-80 direct-solver ceiling);
- observed boundary pressure synthesized on a disjoint **33k-DOF** mesh with **1%**
  complex Gaussian noise, inverted on a disjoint **8.3k-DOF** mesh (no inverse
  crime), 12-38 points per wavelength.

## Dimensional correctness is verified, not assumed

A units bug is easy to hide. `tests/test_radiating.py::test_si_scale_invariance`
proves the SI forward is exact: scaling coordinates by `L` and speeds by `c0`
(holding kR and every speed ratio fixed) must scale the boundary pressure by
exactly `L` and nothing else. The experiment re-checks this live —
**SI/dimensionless forward agreement: max deviation ~6e-12** (i.e. zero).

## A note on optimizer scaling (not a units cheat)

The forward model and its adjoint run entirely in SI. Only the *optimizer's search
variable* is normalized to `u = c / c_bath ~ 1`, with the SI gradient chained by
`dJ/du = c_bath · dJ/dc`. This is ordinary variable scaling: in raw SI the
gradient `dJ/dc ~ 1/c^3` with `c ~ 1500` is so small that L-BFGS-B reads the
objective as flat and stops at iteration zero. Normalizing the *variable* (never
the physics) restores a well-scaled problem and lets the H1 prior use ordinary
order-1 weights.

## Result

| quantity | value |
| --- | --- |
| true tissue speeds | 1465 – 1580 m/s (−2.3% .. +5.3%) |
| recovered speeds | 1462 – 1574 m/s |
| peak contrast recovered | **+4.96%** (true tumour +5.3% → 93% of amplitude) |
| RMS error (interior) | **5.97 m/s = 0.40% of bath speed** |
| relative RMS error | **0.289** |

All three features are recovered in the right place with the right sign: the
tumour sharp and near-full-amplitude, the glandular region broad, and the cyst
resolved despite being only ~0.9 wavelength across at the top frequency. The error
concentrates at feature edges (the highest spatial frequencies, beyond what
kR ≤ 16 can resolve), and the misfit descends an order of magnitude across the
three continuation stages.

**The relative RMS (0.289) matches the dimensionless proof (0.30) — but now at
realistic tissue contrast, in real units, on the faithful open boundary.** The
foundation translates.

## Honest scope

- **Resolution is set by kR, and kR is capped by the direct solver** (`docs/kr_scaling.md`).
  At kR=16 the wavelength is ~3.9 cm, so features below ~2 cm blur together. This
  is a resolution limit, not an error — clinical sub-mm resolution needs the
  high-kR solver work, unchanged by this result.
- Contrast here peaks near 5%; still-weaker or more numerous inclusions are
  harder and would lean more on multi-frequency data and priors.
- Remaining foundation items (`docs/inversion_hypothesis.md`): a measured
  noise-tolerance curve, discrete finite-`N` element drives, and 3D.

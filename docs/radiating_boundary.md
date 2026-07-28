# Physics fidelity II: the radiating (open) boundary

Second physics-fidelity slice, and the more important one: it replaces the model's
one physically *wrong* piece. Built on the verified foundation without modifying
it — `src/usct/radiating.py`, gated by `tests/test_radiating.py`.

## The problem it fixes

The verified DtN forward solves the disk as a **hard-walled cavity** (`u = g` on
the boundary). A cavity has interior resonances: whenever `kR` hits a Bessel zero
(an interior Dirichlet eigenvalue), the response blows up. That is why the
lossless model needed an `eta` stabilizer — `eta` was quietly papering over a
boundary condition that does not describe the real experiment. A real transducer
ring sits in an **open water bath**; outgoing waves radiate away and there are no
such resonances.

## Model

Replace the hard wall with a first-order absorbing / impedance (Robin) condition
carrying the transducer drive `g` (`exp(-i omega t)` convention):

    du/dn - i k_b u = g        on the boundary,   k_b = omega / c_bath

The `-i k_b u` term is the first-order Sommerfeld radiation condition (matched
impedance — outgoing waves leave without reflection); `g` is a velocity-like modal
drive. The natural observable becomes the **boundary pressure** `u` itself — what
a receiving transducer actually reads. This is the **velocity-in / pressure-out**
experiment, the physical counterpart of DtN's pressure-in / flux-out.

Weak form → `A_rad = (K - omega^2 (1 + i eta) M(1/c^2)) - i k_b M_b`, i.e. the
verified volume operator plus one boundary term; the load is `M_b g`. Every DOF
(including the boundary) is an unknown — a full solve, no Dirichlet elimination.
`A_rad` stays complex-symmetric, so the adjoint reuses the forward factorization
(`trans="H"`), exactly as the DtN inversion.

Why no resonances, exactly: the homogeneous-disk boundary pressure is
`u_B = J_m(kR) / (k_b [J_m'(kR) - i J_m(kR)]) * exp(i m theta)`, and the
denominator `J_m'(kR) - i J_m(kR)` never vanishes for real `kR` (`J_m` and `J_m'`
share no zeros). The pole is gone from the math, not merely damped.

## Verification (`tests/test_radiating.py`)

- **Analytic boundary pressure** vs the exact Bessel solution above, modes 0–3:
  `< 2%` mass-relative error (the Bessel analog of the DtN trust test).
- **No interior resonance**: at a DtN resonance `kR = j_{2,1}`, the hard-wall
  cavity field explodes while the radiating field stays `O(1)` — asserted
  `> 50x` smaller.
- **Adjoint gradient** vs central finite differences (lossless and regularized):
  `< 1e-5`.

## Results (`research/04-radiating-boundary/experiment.py`)

**1. Resonances removed at the source.** Sweeping `kR ∈ [3, 12]` for mode `m=2`,
the DtN cavity peaks at `~3.9e3` at exactly the Bessel zeros
(`5.14, 8.42, 11.62`); the radiating boundary stays smooth at `~0.33` with **no
`eta` at all** — an `~11800x` peak ratio. The stabilizer is retired, not hidden.

**2. Reconstruction is not just preserved — it improves.** Recovering a sound-speed
inclusion from **boundary-pressure** data (disjoint mesh + 1% noise) gives relative
RMS error **0.197 — better than the DtN pipeline's 0.30** under the same protocol.
The open problem is better-conditioned precisely because no near-resonant mode
contaminates the data.

## Where this leaves the foundation

- The one physically-wrong piece (closed cavity) is replaced by an open,
  resonance-free experiment that also matches the real velocity-in / pressure-out
  measurement — and it strengthens, rather than costs, the inversion.
- The `eta` scalar can now return to being *physical attenuation*
  (`docs/attenuation.md`), not a resonance band-aid.
- Remaining foundation fidelity items (from `docs/inversion_hypothesis.md`):
  realistic few-percent contrast, an SI (dimensional) end-to-end run, a measured
  noise-tolerance curve, discrete-element (finite `N`) drives, and eventually 3D.
- Exposed as a standalone module (`usct.radiating`), like the other research
  layers; not wired into the CLI, since its observable (boundary pressure) differs
  from the DtN `simulate` path.

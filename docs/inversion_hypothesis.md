# Inversion hypothesis: does adjoint FWI recover c(x) from boundary flux?

**Status: first end-to-end result — the hypothesis works in the idealized 2D
regime.** This note records the method, the verified result, and — just as
importantly — what is *not* yet proven.

This is research layered on top of the verified forward foundation. No verified
forward numerics were modified. The new code is `src/usct/inversion.py` (adjoint
gradient + objective), `src/usct/reconstruction.py` (the L-BFGS-B continuation
driver), and `research/fwi_experiment.py` (the experiment). Correctness is gated
by `tests/test_adjoint.py`.

## The question

Given the verified forward map — boundary pressure `g` in, complex outward flux
`D = I + iQ` out — can we invert it? That is, recover the interior sound speed
`c(x)` from boundary-flux measurements, using an adjoint gradient and low-to-high
frequency continuation, as the project hypothesis (AGENTS.md §1) proposes.

## Method

**Objective (natural boundary inner product).** For predicted flux `D(c)` and
observed flux `D_obs`, per frequency and summed over boundary patterns,

    J(c) = 1/2 (D - D_obs)^H M_b (D - D_obs)  +  alpha/2 * integral |grad c|^2

where `M_b` is the boundary mass matrix, so the data term is exactly the
`L2(boundary)` norm of the flux misfit — mesh-independent, which is what lets
observed and predicted data live on different meshes. `alpha` is an H1
(Tikhonov) smoothness prior.

**Adjoint-state gradient (discretely consistent, factorization reused).** With
`A(m) = K - omega^2 (1 + i eta) M(m)` and nodal squared slowness `m = 1/c^2`,

    lambda_B = D - D_obs                       (adjoint boundary source = residual)
    A_II^H lambda_I = -A_BI^H lambda_B          (SAME interior LU, solved trans="H")
    dJ/dm_j = -omega^2 * Re[(1 + i eta) * integral phi_j conj(lambda) u]
    dJ/dc_j = (-2 / c_j^3) * dJ/dm_j

The one interior factorization that the forward solve builds is re-used for the
adjoint solve (conjugate transpose), so the full gradient over every boundary
pattern costs one extra multi-RHS back-substitution — not a new factorization.
Choosing the `M_b`-weighted misfit is what makes the adjoint boundary source the
*raw* residual.

**Driver.** Bounded L-BFGS-B, low-to-high frequency continuation (each stage
warm-starts from the previous recovered field), adaptive mode count
`m_max = ceil(kR)` per stage, immersion medium (boundary speed) held known.

## Verification (the correctness gate)

`tests/test_adjoint.py` — the analog of the analytic Bessel test, for the
gradient:

- **Central finite differences** vs the adjoint directional derivative agree to
  `< 1e-5` relative error across lossless, lossy (`eta = 0.05`), and regularized
  cases. (The `1 + i eta` term is real bug-bait: an early version assembled the
  gradient load in real arithmetic and silently dropped `Im`, failing only when
  `eta != 0`. Now caught by the lossy case.)
- **Second-order Taylor remainder**: `|J(c+t d) - J(c) - t g.d|` shrinks at order
  → 2 as `t → 0` (a wrong gradient sits pinned at order 1).

## Result

`research/fwi_experiment.py` → `research/outputs/fwi_reconstruction.png`.

Two-inclusion phantom (a `+0.15` "tumor" and a `-0.08` "cyst" on water `c = 1`,
unit disk). Observed flux synthesized on a **33k-DOF mesh** (refinements=7),
1% complex Gaussian noise added, then resampled onto a **disjoint 2.1k-DOF**
inversion mesh (refinements=5) — no inverse crime. Continuation over
`kR = 3, 5, 8`.

| Reconstruction | RMS error (interior) | relative RMS |
| --- | --- | --- |
| homogeneous guess | 0.0597 | 1.00 |
| **frequency continuation** | **0.0178** | **0.30** |
| cold-start highest frequency only | 0.0268 | 0.45 |

Both inclusions are recovered in the correct location, with the correct sign and
approximately correct magnitude; the residual error concentrates only at the
sharp inclusion edges (the highest spatial frequencies). The misfit curve shows
the textbook continuation staircase: each higher-frequency stage injects unfit
data, then descends. Frequency continuation beats a same-budget cold start of the
highest frequency by **33%** RMS and visibly suppresses background speckle — the
cycle-skipping failure mode (AGENTS.md §8), mild at this modest contrast and
expected to become decisive at higher contrast / higher `kR`.

**Conclusion: in the idealized, discretely-consistent 2D DtN regime, the adjoint
FWI hypothesis is confirmed.** The math and the software loop work end to end.

## What this does NOT yet establish (honest scope)

This is a proof of *algorithmic* viability, not of the physical instrument. Load-
bearing gaps, roughly in order of how much they could change the picture:

1. **kR / dimensional reality — now measured (`docs/kr_scaling.md`).**
   Everything here is dimensionless with `kR ≤ 8`; clinically useful resolution
   needs `kR ≈ 400–1300`. The scaling study confirms the direct-solver walls:
   factorization time ~ N^1.38 and, worse, the pollution effect forces DOFs to
   grow like `(kR)^3` to hold accuracy, projecting to ~69 GB and ~9 min per 2D
   factorization at `kR ≈ 400` (3D is out entirely). Practical 2D research
   ceiling on the current direct path is `kR ≈ 50–80`. Reaching clinical `kR`/3D
   means slotting an iterative + preconditioned Helmholtz solver into the
   swappable `Forward` contract — which is exactly what that contract is for.
2. **Attenuation — now a physical field (`docs/attenuation.md`).** `eta` is no
   longer just a stabilizer: it is a spatially-varying attenuation field with a
   verified joint `(c, eta)` adjoint gradient. Physical loss also damps the
   interior resonances (45x at the worst). Open: attenuation imaging is poorly
   conditioned (relative RMS 0.78 vs 0.30 for speed) and wants multi-frequency /
   joint inversion.
3. **DtN vs. a real transducer.** We prescribe pressure and measure flux on a
   continuous aperture. A physical ring prescribes normal velocity from *discrete*
   elements and measures pressure — a physically different boundary experiment,
   and one that caps `m_max` at ~N_elements/2 (spatial Nyquist).
4. **Interior resonances / boundary condition — addressed (`docs/radiating_boundary.md`).**
   The hard-Dirichlet cavity has been replaced by an open, radiating (impedance)
   boundary: analytic-verified, resonances removed at the source (~11800x vs the
   cavity, no `eta`), and it *improves* reconstruction (relative RMS 0.197 vs
   0.30). This is the physically faithful velocity-in / pressure-out experiment.
5. **Higher contrast + real cycle-skipping.** The frozen failure here is mild.
   A high-contrast / high-`kR` case where cold-start fails outright while
   continuation succeeds should be added as a regression fixture.

## Follow-on foundation results

- `docs/attenuation.md` — attenuation as a physical field (verified joint gradient).
- `docs/radiating_boundary.md` — the open/radiating boundary that replaces the
  resonant cavity (analytic-verified, resonances removed, inversion improved).
- `docs/kr_scaling.md` — the direct-solver cost and pollution walls.
- `docs/si_reconstruction.md` — realistic few-percent contrast recovered in real
  SI units on the radiating boundary (relative RMS 0.289; SI exactness verified).

## Reproduce

    python -m pytest tests/test_adjoint.py tests/test_reconstruction.py \
        tests/test_attenuation.py tests/test_radiating.py -q
    python research/fwi_experiment.py           # writes research/outputs/
    python research/radiating_experiment.py
    python research/si_reconstruction.py

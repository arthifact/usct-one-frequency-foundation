# kR scaling: where the direct-solver forward model hits a wall

The reconstruction proof (`docs/inversion_hypothesis.md`) ran at `kR ≤ 8`.
Clinically useful (sub-mm) resolution at a ~10 cm radius needs `kR ≈ 400–1300`.
This study measures the two walls between here and there. Both are intrinsic to
solving the high-frequency Helmholtz equation with a sparse **direct**
factorization — the very thing that makes the adjoint cheap today.

Reproduce: `python research/02-kr-scaling/experiment.py` (writes `research/02-kr-scaling/outputs/kr_scaling.*`).
Dimensionless unit disk, `c = 1`, small loss `eta = 1e-3`; accuracy measured as
the median relative boundary-flux error over a band of angular modes against the
exact complex-argument Bessel solution.

## Wall 1 — cost (hold accuracy, push kR)

Refine the mesh with `kR` to hold ~8 points per wavelength, so accuracy stays
roughly fixed while the problem grows.

| kR | DOFs | factor time | LU nonzeros |
| ---: | ---: | ---: | ---: |
| 5 | 545 | 0.00 s | 0.02 M |
| 20 | 8 321 | 0.03 s | 0.7 M |
| 40 | 33 025 | 0.20 s | 4.4 M |
| 80 | 131 585 | 1.6 s | 23.5 M |
| 100 | 525 313 | 12.3 s | 124 M (~2 GB) |

Measured scaling: **factorization time ~ N^1.38, fill-in ~ N^1.28** (N = DOFs) —
the expected super-linear behaviour of sparse-direct in 2D, on top of DOFs ~ (kR)².

**Projection to clinical `kR ≈ 400` in 2D, same accuracy:** ~8.4M DOFs,
**~9 minutes per factorization, ~69 GB** for the factors (complex128). That is
one frequency, one factorization, in 2D. **3D at that kR is out of reach for
sparse-direct entirely** — there is no cheap nested-dissection ordering in 3D.

## Wall 2 — pollution (fix the mesh, push kR)

The subtler, more important wall. Holding *points per wavelength* constant does
**not** hold the error constant — the finite-element Helmholtz error grows with
`k` (the well-known pollution effect).

Fixed mesh, relative flux error vs kR:

| kR | ppw (r=6, 8k) | error (r=6) | ppw (r=7, 33k) | error (r=7) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 24 | 0.02 | 48 | 0.005 |
| 16 | 12 | 0.20 | 24 | 0.05 |
| 20 | 9.5 | 0.30 | 19 | 0.08 |
| 32 | 6.0 | 1.7 | 12 | 0.7 |
| 40 | 4.8 | 4.0 | 9.5 | 1.0 |

Two things to read off:

1. **At the same points-per-wavelength, higher kR is less accurate.** At ppw ≈
   9.5 the error is 0.30 at kR=20 but 1.0 at kR=40 — quadrupling the mesh to keep
   ppw fixed still let the error grow ~3×.
2. **The 10% error threshold moves slowly.** It sits at kR ≈ 13 on the 8k-DOF
   mesh and only kR ≈ 22 on the 33k-DOF mesh: **4× the DOFs bought ~1.7× the
   usable kR.**

For P1 elements the pollution term scales like `k^3 h^2`, so holding accuracy
requires `h ~ k^-1.5`, i.e. **DOFs ~ (kR)^3, not (kR)^2**. Combined with factor
time ~ DOFs^1.4, wall-clock per factorization grows roughly like `(kR)^4.2`. This
makes Wall 1's projection *optimistic*.

## Consequences for the project

- The batched-LU forward that makes the adjoint nearly free is a **low-frequency,
  2D luxury.** It is excellent for the research regime (`kR ≲ 30–50` in 2D) and
  will not reach clinical `kR` or 3D unchanged.
- This is exactly what the swappable `Forward` contract is for. Reaching high
  `kR` / 3D means slotting in an **iterative solver with a Helmholtz-specific
  preconditioner** (shifted-Laplacian, sweeping, or domain decomposition) behind
  the same interface — provided that interface exposes *"solve many RHS + adjoint"*
  rather than *"LU factor"*. It does.
- Higher-order elements (P2/P3) or a **dispersion-corrected / stabilized** scheme
  would push the pollution wall out substantially and are the cheaper first move
  before changing solvers.
- Practical near-term research ceiling on this machine with the current direct
  path: **kR up to ~50–80 in 2D.** Enough to study contrast, resolution, and
  cycle-skipping; not a clinical instrument.

**Bottom line:** the algorithm is proven; the *forward solver* is the component
that must evolve to scale. Nothing else in the pipeline has to change to do it.

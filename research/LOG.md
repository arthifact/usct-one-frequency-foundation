# Research log

A sequential record of the inversion research: what each step found, and **why it
led to the next**. Each numbered folder here holds the runnable experiment
(`experiment.py`) and its `outputs/`; the detailed write-up for each lives in
`docs/`. Everything is layered on the verified one-frequency forward foundation
**without modifying it**.

Read top to bottom to see the reasoning, not just the results.

---

## 0. Starting point — the hypothesis and the plan

The foundation was a *verified* one-frequency Dirichlet-to-Neumann (DtN) forward
solver: prescribe boundary pressure, measure outward flux, one batched LU over all
patterns, checked against the analytic Bessel disk. The open question: **can we
invert it** — recover the interior sound speed `c(x)` from boundary data?

A review of the proposed plan concluded the real risk was not the data plumbing
but **whether inversion converges**, and that the **adjoint gradient is the
keystone**. Decision: build and *verify* the adjoint first, before anything else.

---

## 1. Adjoint FWI — does the hypothesis work?  → `01-adjoint-fwi/`

**Did:** implemented the complex-L2 boundary-flux objective and its adjoint-state
gradient, reusing the forward LU factorization for the adjoint solve (`trans="H"`).
Gated it with a finite-difference + second-order Taylor test (`tests/test_adjoint.py`).
Then ran an end-to-end reconstruction of a two-inclusion phantom, with data
synthesized on a **disjoint** finer mesh + noise (no inverse crime).

**Result:** the adjoint matches finite differences to `<1e-5` (a real bug surfaced
and was fixed: the complex gradient load was being assembled in real arithmetic,
failing only when loss ≠ 0). Reconstruction recovered both inclusions at relative
RMS **0.30**, and low→high frequency continuation beat a same-budget cold start by
**33%**. **The hypothesis works in 2D.** (`docs/inversion_hypothesis.md`)

**Why next:** the proof ran at low `kR`. What decides whether this ever reaches
*clinical* resolution is how the forward solver scales with frequency — so measure
that wall next.

---

## 2. kR scaling — where does the forward hit a wall?  → `02-kr-scaling/`

**Did:** swept `kR` and measured both the **cost** wall (hold accuracy, grow the
mesh) and the **pollution** wall (fix the mesh, watch accuracy decay).

**Result:** cost — factorization time `~N^1.38`, fill `~N^1.28`; a 2D `kR≈400` run
projects to ~69 GB and ~9 min per factorization, and 3D is out of reach for a
direct solver entirely. Pollution — holding points-per-wavelength fixed does *not*
hold error fixed; for P1 elements the DOFs must grow like `(kR)^3`. **Practical 2D
research ceiling: `kR ≈ 50–80`.** (`docs/kr_scaling.md`)

**Why next:** the takeaway is that only the *forward solver* must eventually
evolve — nothing else in the pipeline. That is a later, heavy job. The user
directed the priority: **hardware/UI is later; make the physics faithful first.**
The most verifiable first fidelity step was attenuation (a clean extension of the
verified adjoint), so do that.

---

## 3. Attenuation — make the loss term physical  → `03-attenuation/`

**Did:** turned the scalar `eta` stabilizer into a spatially-varying physical
attenuation field, with a **verified joint `(c, eta)` adjoint gradient** (one
assembled load yields both gradients). Demonstrated its two consequences.

**Result:** physical attenuation damps the interior cavity resonances (45× at the
worst); attenuation *imaging* localizes an absorber but is poorly conditioned
(relative RMS **0.78** vs 0.30 for speed) — an honest, literature-consistent
"absorption is harder than speed." Crucially, this exposed *why* the resonances
exist: the hard-Dirichlet **cavity**. (`docs/attenuation.md`)

**Why next:** the resonances are a symptom of modeling the ring as a closed cavity.
A real ring sits in an open water bath. That is the one place the model is
physically *wrong* — fix it.

---

## 4. Radiating boundary — the open-medium experiment  → `04-radiating-boundary/`

**Did:** added a new forward *beside* the verified DtN (per AGENTS.md §4): a
first-order impedance/absorbing boundary `∂u/∂n − i k_b u = g`, with **boundary
pressure** as the observable — the physical velocity-in / pressure-out experiment.
Analytic Bessel-verified; adjoint FD-verified.

**Result:** the interior resonances are **removed at the source** (~11,800× smaller
peak than the DtN cavity across `kR`, with **no `eta` at all**), and reconstruction
did not merely survive — it **improved** to relative RMS **0.197 vs 0.30**, the
open problem being better-conditioned. (`docs/radiating_boundary.md`)

**Why next:** with the physics faithful, test whether it *translates to reality* —
real units and the few-percent contrasts real tissue actually shows.

---

## 5. Realistic-contrast SI reconstruction  → `05-si-reconstruction/`

**Did:** reran the reconstruction in **real SI units** (0.10 m disk, water
1500 m/s, frequencies in kHz) at **realistic few-percent tissue contrast**, on the
radiating boundary, with data on a disjoint mesh + noise. Locked dimensional
correctness with a scale-invariance test.

**Result:** recovered all three features; RMS error **5.97 m/s (0.40%)**, relative
RMS **0.289** — matching the dimensionless proof, now at real contrast in real
units. SI exactness verified live to ~6e-12. (One honest gotcha, documented: in
raw SI the gradient `~1/c^3` is so small the optimizer stalls, so the *search
variable* is normalized while the physics stays SI.) (`docs/si_reconstruction.md`)

**Why next:** the cheapest remaining fidelity question with a direct hardware
consequence — how much measurement noise can it take?

---

## 6. Noise-tolerance curve — the hardware SNR requirement  → `06-noise-tolerance/`

**Did:** swept the measurement-noise fraction on the SI radiating setup, multiple
random draws per level, and measured the reconstruction-error curve.

**Result:** the reconstruction is **resolution-limited, not noise-limited** — error
is flat at ~0.29 below ~10% noise, degrades gracefully, and only breaks (error
> 0.5) near **50% noise (~6 dB SNR)**; it is still clearly usable at **20% (~14 dB)**.
So random measurement noise is *not* the binding constraint at this
contrast/resolution — `kR` is. A ~14 dB receiver SNR (easy for real electronics)
suffices. (`docs/noise_tolerance.md`)

**Why next (open):** this covered *random* noise. The error source that actually
threatens real hardware is **systematic** element gain/phase **miscalibration**,
which couples coherently — a calibration-tolerance study is the natural next step.
After that: **discrete finite-N element drives** (element-count vs resolution), and
eventually **3D**. The kR wall (step 2) remains the gate on clinical resolution and
is the one place a new iterative/preconditioned solver must eventually go.

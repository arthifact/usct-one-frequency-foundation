# Physics fidelity I: spatially-varying attenuation

First slice of the physics-fidelity track. Makes attenuation a **physical,
spatially-varying field** (not the scalar `eta` numerical stabilizer) and gives a
**verified adjoint gradient for reconstructing it** alongside sound speed. Built
on the verified foundation without modifying it: `src/usct/attenuation.py`, gated
by `tests/test_attenuation.py`.

## Model

Complex nodal slowness enters the operator `A = K - omega^2 M(s)`:

    s(x) = (1 / c(x)^2) * (1 + i eta(x)),   eta(x) >= 0 (loss tangent)

Physical mapping: `eta = 2 c alpha / omega` with `alpha` the attenuation
coefficient (nepers per unit length). Frequency-linear tissue attenuation
`alpha ~ f` therefore gives a nearly frequency-flat `eta` — unlike the old
`eta ~ omega^2` stabilizer, which was never physical.

`A` remains complex-symmetric, so the adjoint trick survives untouched: the
forward interior LU factorization is re-used for the adjoint solve (`trans="H"`),
and **one** assembled load `L_j = integral phi_j conj(lambda) u` yields *both*
gradients,

    dJ/dm_j   = -omega^2 * Re[(1 + i eta_j) L_j]     (m = 1/c^2 -> sound speed)
    dJ/deta_j =  omega^2 * m_j * Im[L_j]              (-> attenuation)

## Verification

`tests/test_attenuation.py`:
- speed gradient vs central FD (attenuation field present): `< 1e-5`;
- attenuation gradient vs central FD: `< 1e-5`;
- at `eta = 0` the complex path reproduces the verified speed-only adjoint
  (`usct.inversion.objective_and_gradient`) to `1e-8`.

## Results (`research/attenuation_experiment.py`)

**1. Attenuation damps the cavity resonances.** Driving mode `m=2` and sweeping
`kR`, the lossless boundary response spikes at every interior Dirichlet
eigenvalue (`j_{2,1}=5.14`, `j_{2,2}=8.42`, `j_{2,3}=11.6`). A physical loss moves
those eigenvalues off the real axis: `eta=0.02` gives a **45x reduction at the
worst resonance**. This is the honest version of the numerical stabilizer, and it
is why the eventual right answer is a radiating/absorbing boundary (an open water
tank has no such resonances at all).

**2. Attenuation is imageable, but hard.** With sound speed known, reconstructing
an absorption inclusion `eta(x)` from boundary flux (data on a disjoint mesh + 1%
noise) **localizes the absorber in roughly the right place** but is quantitatively
crude: recovered range `[0, 0.12]` vs true `[0.005, 0.065]`, ring artifacts, and
relative RMS error **0.78 — versus 0.30 for sound speed** under the same
protocol. This is expected and honest: absorption couples weakly and diffusely to
the boundary and is poorly conditioned. Multi-frequency data, joint `(c, eta)`
inversion, and stronger priors are the levers to improve it, and are future work.

## Where this leaves physics fidelity

- Done: attenuation is now a physical field with a verified gradient; the
  resonance pathology is understood and quantified.
- Next slice (the other half): an **absorbing / impedance boundary** — modeling
  the ring as sitting in an open (radiating) medium rather than a hard-walled
  cavity. This is a genuinely *different experiment* (interior source in, boundary
  pressure out), so per AGENTS.md section 4 it is added as a **new** forward
  component beside the verified DtN, not folded into it — and it likely wants the
  professor's confirmation of the true source/receiver quantities (AGENTS.md
  section 10) first.
- Still honest scope from `docs/inversion_hypothesis.md`: kR wall, DtN-vs-discrete
  transducer, and 3D remain open.

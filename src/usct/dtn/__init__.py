"""Sim 1 -- Dirichlet-to-Neumann (pressure in -> flux out): the VERIFIED REFERENCE.

DtN is not the physical instrument (a hard-walled cavity has resonances an open
water tank does not). It is kept as the analytically-verifiable gold standard: it
has the cleanest exact Bessel solution, carries the full trust battery
(convergence, outward sign, linearity, Fourier identities, `python -m usct
verify`), and serves as a differential cross-check for the instrument-facing sim.
Build the real instrument path on :mod:`usct.radiating`; check the numerics here.

Depends only on :mod:`usct.physics`; never imports :mod:`usct.radiating`.
"""

# Legacy evidence

`old_lowf_inv.py`, when available, is historical evidence about the professor's
earlier experiment. It is not imported, installed, copied into `src`, or treated
as an architectural dependency.

The new foundation reconstructs only the relevant behavior: circular geometry,
named cosine/sine boundary forcing, a complex Helmholtz forward solve, and boundary
flux observation. Global state, truth-dependent mesh refinement, DOLFINx/PETSc,
Gmsh, process pools, frequency schedules, optimization, adjoints, regularization,
and large plotting blocks are deliberately absent.


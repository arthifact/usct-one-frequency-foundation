# Future research — not part of the foundation

This directory is reserved for hypotheses that may be tested only after the
one-frequency forward simulator remains independently trusted.

Possible later work includes frequency stepping, a conventional complex-L2 inverse
problem, adjoint gradients verified by Taylor tests, phase-difference or
beat-frequency initialization, IR-WRI, optimal-transport objectives, learned
priors, regularization studies, and scalable solvers. GPU acceleration,
matrix-free methods, GMRES, and custom preconditioners should be considered only
after profiling identifies a real limitation.

None of those methods is implemented here. The `src/usct` foundation must never
import from `research`, and future experiments must not change the meaning of the
verified Dirichlet-to-Neumann simulator. If hardware uses discrete sources and
pressure receivers, that model belongs in a separate top-level experiment.


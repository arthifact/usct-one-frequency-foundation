# Mathematics

This note derives the numerical experiment without assuming prior finite-element
experience. The model is deliberately narrow: two dimensions, scalar acoustic
pressure, constant density, isotropic sound speed, and a circular domain with full
boundary access.

## 1. Complex pressure and the time convention

A real sinusoidal pressure is represented by a complex spatial amplitude `u(x)`:

```text
p(x, t) = Re{u(x) exp(-i omega t)}.
```

Here `omega = 2 pi f`, where `f` is in hertz. Complex arithmetic stores the
relative amplitude and phase of every field. With sound speed `c(x)`, squared
slowness is `m(x) = 1/c(x)^2`. The configured loss parameter `eta >= 0` defines

```text
k_squared(x) = omega^2 m(x) (1 + i eta).
```

Loss is never inserted automatically. Lossless verification uses `eta = 0`.

## 2. Boundary-driven Helmholtz problem

The volume equation is

```text
-Laplacian(u) - k_squared(x) u = q  in Omega.
```

The current experiment has no volume source (`q = 0`) and prescribes pressure:

```text
u = g  on boundary(Omega).
```

After solving, it observes the outward normal derivative `du/dn`. Thus the
simulator evaluates a Dirichlet-to-Neumann map: prescribed pressure goes in and
boundary flux comes out. A point source followed by pressure receivers would be a
different experiment.

## 3. Weak form and finite elements

A triangular mesh covers the disk. At every mesh vertex, a first-order Lagrange
basis function `phi_i` equals one at its own vertex, zero at other vertices, and is
linear inside each adjacent triangle. The approximation is

```text
u_h(x) = sum_j U_j phi_j(x).
```

Multiply the PDE by a test basis function `v`, integrate over the domain, and
integrate the Laplacian by parts:

```text
integral_Omega grad(u) dot grad(v) dx
  - integral_Omega k_squared u v dx
  - integral_boundary v du/dn ds
  = integral_Omega q v dx.
```

The volume operator consists of

```text
K_ij = integral_Omega grad(phi_i) dot grad(phi_j) dx,
H_ij = integral_Omega k_squared phi_i phi_j dx,
A = K - H.
```

The code first assembles real geometry and squared-slowness pieces, then combines
them explicitly into `complex128` sparse matrices. Because no complex conjugate is
introduced in the bilinear form, `A` is complex symmetric (`A.T = A`). With
nonzero loss it is generally not Hermitian (`A.conj().T != A`).

## 4. Explicit Dirichlet elimination

Let `B` contain sorted boundary degrees of freedom and `I` contain all interior
degrees of freedom. Put one boundary pattern in each column of `G_B`. Splitting
the matrix equation into blocks gives

```text
A_II U_I + A_IB G_B = Q_I,
A_II U_I = Q_I - A_IB G_B,
U_B = G_B.
```

SciPy factors `A_II` once with sparse direct LU. The entire two-dimensional
right-hand-side array is passed to that single factorization, so every selected
boundary pattern is solved together. For each column the reported normalized
interior residual is

```text
norm(A_II U_I - RHS) / max(norm(RHS), machine_epsilon).
```

## 5. Weak outward Neumann trace

A linear finite-element gradient is constant inside each triangle and jumps
between triangles. Sampling that gradient pointwise on the boundary would produce
a noisy, mesh-dependent trace. Instead, retain the boundary term from the weak
form:

```text
integral_boundary v du/dn ds
  = a(u, v) - integral_Omega q v dx.
```

For a solved coefficient matrix `U`, form its algebraic reaction:

```text
R = A U - Q.
```

The boundary rows `R_B` represent the weak boundary functional. Assemble the
boundary mass matrix

```text
M_boundary_ij = integral_boundary phi_i phi_j ds
```

and solve all columns of

```text
M_boundary_BB D_B = R_B.
```

This is the L2 projection of the outward normal derivative. No empirical sign
change is applied; comparison with the analytic disk fixes the outward sign.

## 6. I/Q views

The primary datum remains complex:

```text
D = I + iQ,
I = real(D),
Q = imag(D),
amplitude = abs(D),
phase = angle(D).
```

Amplitude and phase are derived views. Serialization always retains the original
complex response at `complex128` precision.

## 7. Analytic homogeneous disk

For constant speed, zero loss, radius `R`, mode
`g(theta) = exp(i m theta)`, and `k = omega/c`, separation of variables gives

```text
u_exact(r, theta)
  = J_m(k r) / J_m(k R) exp(i m theta),

du_exact/dn at r=R
  = k J_m'(k R) / J_m(k R) exp(i m theta).
```

Tests use `kR = 3` and modes 0, 1, and 2 after checking that each denominator is
away from zero. For an error vector `e`, the relative finite-element norm is

```text
sqrt(real(conj(e)^T M e) / real(conj(reference)^T M reference)).
```

The domain mass matrix measures field error and the boundary mass matrix measures
flux error. Three successive meshes must reduce both errors substantially.


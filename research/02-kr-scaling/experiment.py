"""kR scaling study: where does the direct-solver forward model hit a wall?

The reconstruction proof ran at kR <= 8. Clinically useful resolution needs
kR of several hundred. This study characterizes the two walls that stand between
here and there, both intrinsic to solving the high-frequency Helmholtz equation
with a sparse *direct* factorization:

  * COST wall -- hold accuracy roughly constant (fixed points-per-wavelength by
    refining the mesh with kR) and watch degrees of freedom, factorization time,
    and LU fill-in grow. 2D FEM DOFs scale like (kR)^2; sparse-direct fill and
    time grow super-linearly on top of that.

  * POLLUTION wall -- fix the mesh and push kR. Even at a fixed number of points
    per wavelength the finite-element Helmholtz error grows with k (the
    "pollution effect"): a fixed mesh that is accurate at low kR silently loses
    accuracy at high kR.

Accuracy is measured against the exact homogeneous-disk Bessel flux -- the same
reference the verified forward tests use, here parameterized by kR.

Run:  python research/kr_scaling.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2, pi
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402
from scipy.special import jv, jvp  # noqa: E402

from usct.domain import build_disk  # noqa: E402
from usct.forcing import build_boundary_forcing  # noqa: E402
from usct.measurement import boundary_mass_matrix  # noqa: E402
from usct.medium import constant_speed  # noqa: E402
from usct.operator import assemble_helmholtz  # noqa: E402

MAX_REFINEMENT = 9  # r=10 (~2.1M dofs) is the practical laptop wall; stop before it
DOF_CAP = 600_000


LOSS = 1e-3  # small stabilizer against interior Dirichlet resonances (as inversion uses)


@dataclass(frozen=True)
class Measurement:
    k_radius: float
    refinements: int
    dofs: int
    boundary_dofs: int
    points_per_wavelength: float
    mode_count: int
    assemble_seconds: float
    factor_seconds: float
    solve_seconds: float
    fill_nonzeros: int
    flux_error: float


def _mode_band(k_radius: float, boundary_count: int) -> list[int]:
    """A representative band of angular modes up to ~0.7 kR.

    Reporting the *median* error over a band (rather than one mode) keeps the
    accuracy metric from being dominated by a single near-resonant mode -- the
    small loss keeps every mode's Bessel coefficient finite -- and reflects that
    imaging actually uses the whole band up to m ~ kR.
    """

    top = max(2, min(round(0.7 * k_radius), boundary_count // 4))
    return sorted({int(round(value)) for value in np.linspace(1, top, 6) if round(value) >= 1})


def _relative_error(numerical, reference, mass) -> float:
    error = numerical - reference
    numerator = float(np.real(np.vdot(error, mass @ error)))
    denominator = float(np.real(np.vdot(reference, mass @ reference)))
    return float(np.sqrt(max(numerator, 0.0) / denominator))


def measure(k_radius: float, refinements: int) -> Measurement:
    """One forward solve of the homogeneous disk at ``kR``; times cost and error.

    Dimensionless: radius = 1, c = 1, so omega = kR and the (lossy) wavenumber is
    ``k_eff = omega sqrt(1 + i eta)``. A whole band of angular modes is solved
    together from a single interior LU factorization (timed, with its fill
    counted); the reported flux error is the median relative error over the band,
    each compared to its exact complex-argument Bessel flux.
    """

    domain = build_disk(1.0, refinements)
    omega = k_radius  # R = 1, c = 1
    wavenumber = omega * np.sqrt(1.0 + 1j * LOSS)
    medium = constant_speed(domain, 1.0)

    started = perf_counter()
    operator = assemble_helmholtz(domain, medium, omega, LOSS)
    assemble_seconds = perf_counter() - started

    matrix = operator.matrix.tocsc()
    interior = domain.interior_dofs
    boundary = domain.boundary_dofs
    a_ii = matrix[interior][:, interior].tocsc()
    a_ib = matrix[interior][:, boundary]

    modes = _mode_band(k_radius, boundary.size)
    forcing = build_boundary_forcing(domain, tuple(f"exp:{m}" for m in modes))
    g = forcing.values

    started = perf_counter()
    factor = splu(a_ii)
    factor_seconds = perf_counter() - started
    fill = int(factor.L.nnz + factor.U.nnz)

    started = perf_counter()
    interior_field = np.asarray(factor.solve(-(a_ib @ g)), dtype=np.complex128)
    solve_seconds = perf_counter() - started
    if interior_field.ndim == 1:
        interior_field = interior_field[:, None]
    field = np.empty((domain.basis.N, len(modes)), dtype=np.complex128)
    field[interior] = interior_field
    field[boundary] = g

    reaction = np.asarray((matrix @ field)[boundary], dtype=np.complex128)
    boundary_mass = boundary_mass_matrix(domain)
    predicted = np.asarray(splu(boundary_mass.astype(np.complex128)).solve(reaction))
    if predicted.ndim == 1:
        predicted = predicted[:, None]

    theta = domain.boundary_theta
    errors = []
    for column, mode in enumerate(modes):
        coefficient = wavenumber * jvp(mode, wavenumber) / jv(mode, wavenumber)
        exact_flux = coefficient * np.exp(1j * mode * theta)
        errors.append(_relative_error(predicted[:, column], exact_flux, boundary_mass))

    wavelength = 2.0 * pi / omega
    return Measurement(
        k_radius=k_radius,
        refinements=refinements,
        dofs=int(domain.basis.N),
        boundary_dofs=int(boundary.size),
        points_per_wavelength=wavelength / domain.h_max,
        mode_count=len(modes),
        assemble_seconds=assemble_seconds,
        factor_seconds=factor_seconds,
        solve_seconds=solve_seconds,
        fill_nonzeros=fill,
        flux_error=float(np.median(errors)),
    )


def refinement_for(k_radius: float, target_points: float) -> int:
    """Smallest refinement holding ~``target_points`` per wavelength (h ~ 2.06/2^r)."""

    needed = target_points * k_radius * 2.06 / (2.0 * pi)
    return int(min(MAX_REFINEMENT, max(3, ceil(log2(needed)))))


def cost_wall_sweep(target_points: float = 8.0):
    print(f"\n=== COST WALL: hold ~{target_points:.0f} points/wavelength, push kR ===")
    header = f"{'kR':>6} {'r':>3} {'dofs':>9} {'ppw':>6} {'modes':>6} " \
             f"{'factor s':>9} {'solve s':>8} {'fill M':>8} {'flux err':>10}"
    print(header)
    rows = []
    for k_radius in (5, 10, 15, 20, 30, 40, 60, 80, 100):
        r = refinement_for(k_radius, target_points)
        expected = build_disk(1.0, r).basis.N
        if expected > DOF_CAP:
            print(f"{k_radius:>6.0f} {r:>3} {'>cap':>9}  (skipped: {expected} dofs > {DOF_CAP})")
            continue
        m = measure(float(k_radius), r)
        rows.append(m)
        print(f"{m.k_radius:>6.0f} {m.refinements:>3} {m.dofs:>9} {m.points_per_wavelength:>6.1f} "
              f"{m.mode_count:>6} {m.factor_seconds:>9.2f} {m.solve_seconds:>8.3f} "
              f"{m.fill_nonzeros/1e6:>8.2f} {m.flux_error:>10.2e}")
    return rows


def pollution_sweep(refinements: int):
    print(f"\n=== POLLUTION WALL: fix mesh at r={refinements}, push kR ===")
    domain_dofs = build_disk(1.0, refinements).basis.N
    print(f"fixed mesh: {domain_dofs} dofs")
    print(f"{'kR':>6} {'ppw':>7} {'modes':>6} {'flux err':>10}")
    rows = []
    for k_radius in (2, 3, 5, 8, 12, 16, 20, 26, 32, 40):
        m = measure(float(k_radius), refinements)
        rows.append(m)
        print(f"{m.k_radius:>6.0f} {m.points_per_wavelength:>7.1f} "
              f"{m.mode_count:>6} {m.flux_error:>10.2e}")
    return rows


def make_figure(cost_rows, pollution_sets, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

    kr = np.array([m.k_radius for m in cost_rows])
    dofs = np.array([m.dofs for m in cost_rows])
    factor = np.array([m.factor_seconds for m in cost_rows])
    fill = np.array([m.fill_nonzeros for m in cost_rows])

    axes[0].loglog(kr, dofs, "o-", label="measured DOFs")
    axes[0].loglog(kr, dofs[0] * (kr / kr[0]) ** 2, "--", color="gray", label="(kR)^2")
    axes[0].set_title("DOFs vs kR (fixed accuracy)")
    axes[0].set_xlabel("kR")
    axes[0].set_ylabel("degrees of freedom")
    axes[0].legend()

    axes[1].loglog(dofs, factor, "o-", label="factorization time")
    axes[1].loglog(dofs, factor[0] * (dofs / dofs[0]) ** 1.5, "--", color="gray",
                   label="O(N^1.5)")
    axes[1].set_title("Direct-solve factorization cost")
    axes[1].set_xlabel("degrees of freedom N")
    axes[1].set_ylabel("factor time (s)")
    axes[1].legend()

    axes[2].loglog(dofs, fill, "o-", label="LU nonzeros")
    axes[2].loglog(dofs, fill[0] * (dofs / dofs[0]), "--", color="gray", label="O(N)")
    axes[2].set_title("LU fill-in (memory)")
    axes[2].set_xlabel("degrees of freedom N")
    axes[2].set_ylabel("nonzeros in L+U")
    axes[2].legend()

    for refinements, rows in pollution_sets:
        pk = np.array([m.k_radius for m in rows])
        perr = np.array([m.flux_error for m in rows])
        axes[3].semilogy(pk, perr, "o-", label=f"fixed r={refinements} ({rows[0].dofs} dofs)")
    axes[3].axhline(0.1, color="crimson", ls=":", label="10% flux error")
    axes[3].set_title("Pollution: fixed mesh loses accuracy with kR")
    axes[3].set_xlabel("kR")
    axes[3].set_ylabel("relative flux error")
    axes[3].legend()

    fig.suptitle("kR scaling: cost and pollution walls of the direct-solver forward model",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def wall_analysis(cost_rows):
    print("\n=== Wall analysis ===")
    dofs = np.array([m.dofs for m in cost_rows], dtype=float)
    factor = np.array([m.factor_seconds for m in cost_rows], dtype=float)
    fill = np.array([m.fill_nonzeros for m in cost_rows], dtype=float)
    time_slope = np.polyfit(np.log(dofs), np.log(np.maximum(factor, 1e-6)), 1)[0]
    fill_slope = np.polyfit(np.log(dofs), np.log(fill), 1)[0]
    print(f"factor time ~ N^{time_slope:.2f}   fill ~ N^{fill_slope:.2f}   (N = DOFs)")
    top = cost_rows[-1]
    # Project to a clinical kR ~ 400 in 2D at the same points/wavelength:
    #   DOFs ~ (kR)^2, so relative to the top row measured here.
    clinical_kr = 400.0
    dof_projection = top.dofs * (clinical_kr / top.k_radius) ** 2
    time_projection = top.factor_seconds * (dof_projection / top.dofs) ** time_slope
    fill_projection = top.fill_nonzeros * (dof_projection / top.dofs) ** fill_slope
    print(f"top measured point: kR={top.k_radius:.0f}, {top.dofs} dofs, "
          f"{top.factor_seconds:.1f}s, {top.fill_nonzeros/1e6:.0f}M nonzeros")
    print(f"projection to clinical kR~{clinical_kr:.0f} in 2D at the same accuracy:")
    print(f"   ~{dof_projection/1e6:.1f}M dofs, ~{time_projection:.0f}s per factorization, "
          f"~{fill_projection/1e6:.0f}M nonzeros (~{fill_projection*16/1e9:.0f} GB, complex128)")
    print("   3D at the same kR is out of reach for sparse-direct entirely; the")
    print("   swappable Forward contract is where an iterative + preconditioned")
    print("   solver (shifted-Laplacian / sweeping / domain decomposition) slots in.")


def main():
    cost_rows = cost_wall_sweep(target_points=8.0)
    pollution_low = pollution_sweep(refinements=6)
    pollution_high = pollution_sweep(refinements=7)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    figure_path = output_dir / "kr_scaling.png"
    make_figure(cost_rows, [(6, pollution_low), (7, pollution_high)], figure_path)
    wall_analysis(cost_rows)

    np.savez_compressed(
        output_dir / "kr_scaling.npz",
        cost_kr=np.array([m.k_radius for m in cost_rows]),
        cost_dofs=np.array([m.dofs for m in cost_rows]),
        cost_factor_seconds=np.array([m.factor_seconds for m in cost_rows]),
        cost_fill=np.array([m.fill_nonzeros for m in cost_rows]),
        cost_flux_error=np.array([m.flux_error for m in cost_rows]),
    )
    print(f"\nfigure -> {figure_path}")


if __name__ == "__main__":
    main()

"""Radiating boundary: resonances gone (not damped), and still fully invertible.

Two demonstrations built on the verified radiating forward (`src/usct/radiating.py`):

1. NO RESONANCES. Sweep kR across the interior Dirichlet eigenvalues (Bessel
   zeros). The hard-walled DtN cavity spikes at every one; the radiating boundary
   -- lossless, no eta stabilizer -- stays smooth and bounded. The resonances are
   removed at the source, not damped by an artificial loss.

2. RECONSTRUCTION FROM BOUNDARY PRESSURE. Recover a sound-speed inclusion from the
   radiating experiment's natural observable (boundary pressure, driven by
   velocity-like modal sources), with data on a disjoint mesh + 1% noise. Confirms
   the physically faithful forward carries inversion end to end.

Run:  python research/radiating_experiment.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import jn_zeros  # noqa: E402

from usct.domain import build_disk  # noqa: E402
from usct.forcing import build_boundary_forcing  # noqa: E402
from usct.inversion import resample_flux_to  # noqa: E402
from usct.measurement import boundary_mass_matrix  # noqa: E402
from usct.medium import constant_speed, feature_speed  # noqa: E402
from usct.operator import assemble_helmholtz  # noqa: E402
from usct.radiating import (  # noqa: E402
    forward_boundary_pressure,
    full_boundary_mass,
    radiating_objective_and_gradient,
    solve_radiating,
)
from usct.solver import solve_dirichlet  # noqa: E402

BATH_SPEED = 1.0


def resonance_sweep():
    """Peak boundary response vs kR: DtN cavity (spikes) vs radiating (smooth)."""

    domain = build_disk(1.0, refinements=6)
    mode = 2
    forcing = build_boundary_forcing(domain, (f"exp:{mode}",))
    speed = np.ones(domain.basis.N)
    medium = constant_speed(domain, 1.0)
    k_values = np.linspace(3.0, 12.0, 200)
    dtn, radiating = [], []
    for k_radius in k_values:
        operator = assemble_helmholtz(domain, medium, k_radius, 0.0)
        dtn.append(float(np.max(np.abs(solve_dirichlet(domain, operator, forcing).field))))
        solution = solve_radiating(domain, speed, forcing, k_radius, 0.0, BATH_SPEED)
        radiating.append(float(np.max(np.abs(solution.field))))
    resonances = [z for z in jn_zeros(mode, 4) if k_values[0] <= z <= k_values[-1]]
    return k_values, np.asarray(dtn), np.asarray(radiating), resonances


def reconstruction():
    """Recover a sound-speed inclusion from boundary-pressure data (disjoint mesh)."""

    observation_domain = build_disk(1.0, refinements=6)
    inversion_domain = build_disk(1.0, refinements=5)
    rng = np.random.default_rng(11)

    features = ({"type": "disk", "label": "inclusion", "center": (0.2, -0.15),
                 "radius": 0.28, "contrast": 0.12},)
    truth = feature_speed(observation_domain, BATH_SPEED, 18.0, features).sound_speed

    frequency = 8.0 / (2 * np.pi)
    omega = 2.0 * np.pi * frequency
    patterns = tuple(["constant"] + [p for m in range(1, 9) for p in (f"cos:{m}", f"sin:{m}")])
    forcing_obs = build_boundary_forcing(observation_domain, patterns)
    clean = forward_boundary_pressure(
        observation_domain, truth, forcing_obs, omega, 0.0, BATH_SPEED
    )
    rms = np.sqrt(np.mean(np.abs(clean) ** 2))
    noise = rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    noisy = clean + 0.01 * rms / np.sqrt(2.0) * noise
    observed = resample_flux_to(
        inversion_domain.boundary_theta, observation_domain.boundary_theta, noisy
    )

    forcing_inv = build_boundary_forcing(inversion_domain, patterns)
    boundary_mass_full = full_boundary_mass(inversion_domain)
    boundary_mass = boundary_mass_matrix(inversion_domain)
    boundary = inversion_domain.boundary_dofs
    history: list[float] = []

    def objective(candidate):
        speed = candidate.copy()
        speed[boundary] = BATH_SPEED
        result = radiating_objective_and_gradient(
            inversion_domain, speed, forcing_inv, omega, 0.0, BATH_SPEED, observed,
            alpha=1e-4, background_speed=BATH_SPEED,
            boundary_mass_full=boundary_mass_full, boundary_mass=boundary_mass,
        )
        history.append(result.data_misfit)
        return result.misfit, result.gradient

    initial = np.full(inversion_domain.basis.N, BATH_SPEED)
    bounds = [(0.85, 1.25)] * inversion_domain.basis.N
    for dof in boundary:
        bounds[int(dof)] = (BATH_SPEED, BATH_SPEED)
    optimized = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 60, "ftol": 1e-12, "gtol": 1e-9})

    recovered = optimized.x
    truth_inv = feature_speed(inversion_domain, BATH_SPEED, 18.0, features).sound_speed
    interior = inversion_domain.interior_dofs
    error = recovered[interior] - truth_inv[interior]
    rms_error = float(np.sqrt(np.mean(error**2)))
    baseline = float(np.sqrt(np.mean((truth_inv[interior] - BATH_SPEED) ** 2)))
    return {
        "domain": inversion_domain, "truth": truth_inv, "recovered": recovered,
        "history": history, "rms_error": rms_error, "relative_rms_error": rms_error / baseline,
        "recovered_range": (float(recovered.min()), float(recovered.max())),
    }


def make_figure(sweep, recon, output_path):
    k_values, dtn, radiating, resonances = sweep
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

    axes[0].semilogy(k_values, dtn, lw=1.5, label="DtN cavity (hard wall)")
    axes[0].semilogy(k_values, radiating, lw=1.8, label="radiating (open)")
    for zero in resonances:
        axes[0].axvline(zero, color="crimson", ls=":", alpha=0.6)
    axes[0].set_title("Interior resonances: cavity vs open (mode m=2)")
    axes[0].set_xlabel("kR")
    axes[0].set_ylabel("max |field|")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    domain = recon["domain"]
    x, y, tri = domain.coordinates[:, 0], domain.coordinates[:, 1], domain.triangles
    for ax, values, title in (
        (axes[1], recon["truth"], "Truth c(x)"),
        (axes[2], recon["recovered"], "Recovered c(x) (boundary pressure)"),
    ):
        tpc = ax.tricontourf(x, y, tri, values, levels=24, cmap="inferno", vmin=0.85, vmax=1.20)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    axes[3].semilogy(np.arange(len(recon["history"])), recon["history"], lw=1.5)
    axes[3].set_title("Reconstruction misfit")
    axes[3].set_xlabel("objective evaluation")
    axes[3].set_ylabel("data misfit")
    axes[3].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Radiating boundary: resonances removed at the source, and still fully invertible",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    print("=== Resonance sweep (mode m=2, kR in [3,12]) ===")
    sweep = resonance_sweep()
    _, dtn, radiating, resonances = sweep
    print(f"DtN cavity peak:  {dtn.max():.2e}  (spikes at Bessel zeros {np.round(resonances, 3)})")
    print(f"radiating peak:   {radiating.max():.2e}  (smooth, no eta stabilizer)")
    print(f"cavity/radiating peak ratio: {dtn.max() / radiating.max():.0f}x")

    print("\n=== Reconstruction from boundary pressure ===")
    recon = reconstruction()
    print(f"recovered c range: [{recon['recovered_range'][0]:.4f}, "
          f"{recon['recovered_range'][1]:.4f}]")
    print(f"RMS error: {recon['rms_error']:.5f}  (relative {recon['relative_rms_error']:.3f})")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    figure_path = output_dir / "radiating.png"
    make_figure(sweep, recon, figure_path)
    print(f"\nfigure -> {figure_path}")


if __name__ == "__main__":
    main()

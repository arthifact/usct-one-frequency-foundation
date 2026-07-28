"""Noise-tolerance curve: how much measurement noise can the reconstruction take?

Replaces the "1% noise worked once" anecdote with a measured breakdown curve --
which is exactly the hardware SNR requirement. It sweeps the measurement-noise
fraction on the realistic-contrast SI radiating-boundary setup (the same phantom
and continuation as `si_reconstruction.py`), several random noise draws per level,
and records how the reconstruction error grows. The clean boundary data is
computed once and reused; only fresh noise is added per (level, seed).

Run:  python research/noise_tolerance.py
"""

from __future__ import annotations

from math import ceil, pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from skfem import asm  # noqa: E402

from usct.physics.boundary import (
    boundary_mass_matrix,  # noqa: E402
    resample_flux_to,  # noqa: E402
)
from usct.physics.domain import build_disk  # noqa: E402
from usct.physics.forcing import build_boundary_forcing  # noqa: E402
from usct.physics.medium import feature_speed  # noqa: E402
from usct.physics.operator import _stiffness_form  # noqa: E402
from usct.radiating.forward import forward_boundary_pressure, full_boundary_mass  # noqa: E402
from usct.radiating.inversion import radiating_objective_and_gradient  # noqa: E402

RADIUS = 0.10
BATH_SPEED = 1500.0
SPEED_BOUNDS = (1400.0, 1600.0)
TRUTH_FEATURES = (
    {"type": "ellipse", "label": "gland", "center": (0.0, 0.0),
     "semi_axes": (0.045, 0.035), "contrast": 25.0},
    {"type": "disk", "label": "tumour", "center": (0.035, -0.02),
     "radius": 0.020, "contrast": 55.0},
    {"type": "disk", "label": "cyst", "center": (-0.03, 0.03),
     "radius": 0.018, "contrast": -35.0},
)
TRUTH_SHARPNESS = 250.0
K_RADII = (5.0, 10.0, 16.0)
ALPHAS = (3e-5, 1e-5, 3e-6)
MAX_ITERATIONS = 40

NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
SEEDS = (0, 1)
IMAGE_LEVELS = (0.0, 0.20, 1.0)  # recovered fields to display (seed 0)


def frequency_for(k_radius):
    return k_radius * BATH_SPEED / (2.0 * pi * RADIUS)


def patterns_for(k_radius):
    max_mode = max(2, ceil(k_radius))
    modes = [p for m in range(1, max_mode + 1) for p in (f"cos:{m}", f"sin:{m}")]
    return tuple(["constant"] + modes)


def truth_speed(domain):
    return feature_speed(domain, BATH_SPEED, TRUTH_SHARPNESS, TRUTH_FEATURES).sound_speed


def clean_data(observation_domain, inversion_domain):
    """Noise-free boundary pressure per stage, plus its RMS (for scaling noise)."""

    truth = truth_speed(observation_domain)
    stages = []
    for k_radius in K_RADII:
        forcing = build_boundary_forcing(observation_domain, patterns_for(k_radius))
        omega = 2.0 * pi * frequency_for(k_radius)
        clean = forward_boundary_pressure(
            observation_domain, truth, forcing, omega, 0.0, BATH_SPEED
        )
        stages.append((clean, float(np.sqrt(np.mean(np.abs(clean) ** 2)))))
    return stages


def reconstruct(inversion_domain, observed, caches):
    """Frequency-continuation FWI in SI over the normalized variable u = c/bath."""

    boundary_mass_full, boundary_mass, stiffness = caches
    boundary = inversion_domain.boundary_dofs
    normalized = np.ones(inversion_domain.basis.N)
    low, high = SPEED_BOUNDS[0] / BATH_SPEED, SPEED_BOUNDS[1] / BATH_SPEED
    bounds = [(low, high)] * inversion_domain.basis.N
    for dof in boundary:
        bounds[int(dof)] = (1.0, 1.0)

    for index, k_radius in enumerate(K_RADII):
        forcing = build_boundary_forcing(inversion_domain, patterns_for(k_radius))
        omega = 2.0 * pi * frequency_for(k_radius)
        observed_flux = observed[index]
        alpha = ALPHAS[index]

        def evaluate(candidate, _forcing=forcing, _omega=omega,
                     _observed=observed_flux, _alpha=alpha):
            speed = candidate * BATH_SPEED
            result = radiating_objective_and_gradient(
                inversion_domain, speed, _forcing, _omega, 0.0, BATH_SPEED, _observed,
                alpha=0.0, background_speed=BATH_SPEED,
                boundary_mass_full=boundary_mass_full, boundary_mass=boundary_mass,
            )
            gradient = result.gradient * BATH_SPEED
            stiffness_u = np.asarray(stiffness @ candidate, dtype=np.float64)
            gradient = gradient + _alpha * stiffness_u
            gradient[boundary] = 0.0
            reg = 0.5 * _alpha * float(candidate @ stiffness_u)
            return result.data_misfit + reg, gradient

        optimized = minimize(evaluate, normalized, jac=True, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": MAX_ITERATIONS, "ftol": 1e-14, "gtol": 1e-12})
        normalized = np.clip(optimized.x, low, high)
    return normalized * BATH_SPEED


def relative_rms(inversion_domain, recovered, truth):
    interior = inversion_domain.interior_dofs
    error = recovered[interior] - truth[interior]
    baseline = np.sqrt(np.mean((truth[interior] - BATH_SPEED) ** 2))
    return float(np.sqrt(np.mean(error**2)) / baseline)


def run_sweep(observation_domain, inversion_domain):
    stages_clean = clean_data(observation_domain, inversion_domain)
    truth_inv = truth_speed(inversion_domain)
    caches = (
        full_boundary_mass(inversion_domain),
        boundary_mass_matrix(inversion_domain),
        asm(_stiffness_form, inversion_domain.basis).tocsc(),
    )

    results: dict[float, list[float]] = {level: [] for level in NOISE_LEVELS}
    images: dict[float, np.ndarray] = {}
    for level in NOISE_LEVELS:
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            observed = {}
            for index, (clean, rms) in enumerate(stages_clean):
                noise = rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
                noisy = clean + level * rms / np.sqrt(2.0) * noise
                observed[index] = resample_flux_to(
                    inversion_domain.boundary_theta, observation_domain.boundary_theta, noisy
                )
            recovered = reconstruct(inversion_domain, observed, caches)
            results[level].append(relative_rms(inversion_domain, recovered, truth_inv))
            if seed == 0 and level in IMAGE_LEVELS:
                images[level] = recovered
            if level == 0.0:
                break  # no randomness at zero noise
        print(f"  noise {level*100:5.1f}%: relative RMS "
              f"{np.mean(results[level]):.3f} "
              f"(min {np.min(results[level]):.3f}, max {np.max(results[level]):.3f})")
    return results, images, truth_inv


def make_figure(inversion_domain, results, images, truth, output_path):
    levels = np.array(NOISE_LEVELS) * 100.0
    means = np.array([np.mean(results[level]) for level in NOISE_LEVELS])
    lows = np.array([np.min(results[level]) for level in NOISE_LEVELS])
    highs = np.array([np.max(results[level]) for level in NOISE_LEVELS])

    fig, axes = plt.subplots(1, 5, figsize=(23, 4.4))
    axes[0].fill_between(levels, lows, highs, alpha=0.25, color="steelblue",
                         label="min-max over seeds")
    axes[0].plot(levels, means, "o-", color="steelblue", lw=1.8, label="mean relative RMS")
    axes[0].axhline(1.0, color="crimson", ls=":", label="homogeneous (no recovery)")
    axes[0].set_title("Noise tolerance")
    axes[0].set_xlabel("measurement noise (% of RMS boundary pressure)")
    axes[0].set_ylabel("relative RMS error")
    axes[0].set_ylim(0, 1.15)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    x = inversion_domain.coordinates[:, 0]
    y = inversion_domain.coordinates[:, 1]
    tri = inversion_domain.triangles
    panels = [(axes[1], truth, "Truth c(x)  [m/s]")]
    for offset, level in enumerate(IMAGE_LEVELS):
        panels.append((axes[2 + offset], images[level],
                       f"Recovered @ {level*100:.0f}% noise  [m/s]"))
    for ax, values, title in panels:
        tpc = ax.tricontourf(x, y, tri, values, levels=24, cmap="inferno", vmin=1460, vmax=1560)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Noise-tolerance curve: reconstruction error vs measurement noise "
        "(0.10 m disk, water 1500 m/s, tissue contrast <5%)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    observation_domain = build_disk(RADIUS, refinements=7)
    inversion_domain = build_disk(RADIUS, refinements=6)
    print("=== Noise-tolerance sweep (radiating boundary, SI, realistic contrast) ===")
    print(f"observation mesh {observation_domain.basis.N} dofs, inversion mesh "
          f"{inversion_domain.basis.N} dofs; {len(SEEDS)} seeds per level")

    results, images, truth = run_sweep(observation_domain, inversion_domain)

    # Locate the breakdown knee: first level whose mean error exceeds 0.5.
    knee = None
    for level in NOISE_LEVELS:
        if np.mean(results[level]) > 0.5:
            knee = level
            break
    print("\n=== Summary ===")
    baseline = np.mean(results[0.0])
    print(f"noise-free relative RMS: {baseline:.3f}")
    if knee is not None:
        print(f"breakdown (mean relative RMS > 0.5) first at {knee*100:.1f}% noise")
    else:
        print("no breakdown up to the maximum swept noise level")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    figure_path = output_dir / "noise_tolerance.png"
    make_figure(inversion_domain, results, images, truth, figure_path)
    np.savez_compressed(
        output_dir / "noise_tolerance.npz",
        noise_levels=np.array(NOISE_LEVELS),
        relative_rms_mean=np.array([np.mean(results[level]) for level in NOISE_LEVELS]),
        relative_rms_min=np.array([np.min(results[level]) for level in NOISE_LEVELS]),
        relative_rms_max=np.array([np.max(results[level]) for level in NOISE_LEVELS]),
    )
    print(f"\nfigure -> {figure_path}")


if __name__ == "__main__":
    main()

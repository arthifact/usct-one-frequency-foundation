"""Calibration-tolerance curve: how much systematic element error can we take?

The noise-tolerance study (step 06) found random measurement noise is NOT the
binding constraint -- so what is? The prime suspect is SYSTEMATIC per-element
miscalibration: each transducer has an unknown complex gain (amplitude + phase)
error that is FIXED, and therefore consistent across every frequency and every
drive pattern. Unlike random noise (independent per measurement, averages out),
a fixed gain error couples coherently and the inversion cannot average it away.

Model (receive-side calibration): each boundary element j has a complex gain
`g_j = (1 + a_j) exp(i phi_j)` with `a_j ~ N(0, level)` (amplitude) and
`phi_j ~ N(0, level rad)` (phase), drawn ONCE per device (seed) and applied to the
observed boundary pressure at every stage. The inversion uses the nominal
(g = 1) forward, so the miscalibration is an unmodeled, coherent data error.

Built on the instrument-facing radiating sim, same phantom/continuation as step
05/06. Overlays the step-06 noise curve for direct contrast if available.

Run:  python research/07-calibration-tolerance/experiment.py
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

from usct.physics.boundary import boundary_mass_matrix, resample_flux_to  # noqa: E402
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

CALIBRATION_LEVELS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)
SEEDS = (0, 1)  # independent "devices"
IMAGE_LEVELS = (0.0, 0.05, 0.20)


def frequency_for(k_radius):
    return k_radius * BATH_SPEED / (2.0 * pi * RADIUS)


def patterns_for(k_radius):
    max_mode = max(2, ceil(k_radius))
    modes = [p for m in range(1, max_mode + 1) for p in (f"cos:{m}", f"sin:{m}")]
    return tuple(["constant"] + modes)


def truth_speed(domain):
    return feature_speed(domain, BATH_SPEED, TRUTH_SHARPNESS, TRUTH_FEATURES).sound_speed


def clean_data(observation_domain):
    """Noise-free boundary pressure per stage on the observation mesh."""

    truth = truth_speed(observation_domain)
    stages = []
    for k_radius in K_RADII:
        forcing = build_boundary_forcing(observation_domain, patterns_for(k_radius))
        omega = 2.0 * pi * frequency_for(k_radius)
        clean = forward_boundary_pressure(
            observation_domain, truth, forcing, omega, 0.0, BATH_SPEED
        )
        stages.append(clean)
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


def element_gain(observation_domain, level, seed):
    """Fixed per-element complex gain error (drawn once per device)."""

    rng = np.random.default_rng(2000 + seed)
    n = observation_domain.boundary_dofs.size
    amplitude = 1.0 + level * rng.standard_normal(n)
    phase = level * rng.standard_normal(n)  # radians
    return amplitude * np.exp(1j * phase)


def run_sweep(observation_domain, inversion_domain):
    stages_clean = clean_data(observation_domain)
    truth_inv = truth_speed(inversion_domain)
    caches = (
        full_boundary_mass(inversion_domain),
        boundary_mass_matrix(inversion_domain),
        asm(_stiffness_form, inversion_domain.basis).tocsc(),
    )

    results: dict[float, list[float]] = {level: [] for level in CALIBRATION_LEVELS}
    images: dict[float, np.ndarray] = {}
    for level in CALIBRATION_LEVELS:
        for seed in SEEDS:
            gain = element_gain(observation_domain, level, seed)
            observed = {}
            for index, clean in enumerate(stages_clean):
                corrupted = gain[:, None] * clean  # same gain across all stages
                observed[index] = resample_flux_to(
                    inversion_domain.boundary_theta, observation_domain.boundary_theta, corrupted
                )
            recovered = reconstruct(inversion_domain, observed, caches)
            results[level].append(relative_rms(inversion_domain, recovered, truth_inv))
            if seed == 0 and level in IMAGE_LEVELS:
                images[level] = recovered
            if level == 0.0:
                break  # no error at level 0
        print(f"  calib {level*100:5.1f}% ({np.degrees(level):4.1f} deg phase): "
              f"relative RMS {np.mean(results[level]):.3f} "
              f"(min {np.min(results[level]):.3f}, max {np.max(results[level]):.3f})")
    return results, images, truth_inv


def load_noise_curve():
    path = (Path(__file__).resolve().parent.parent
            / "06-noise-tolerance" / "outputs" / "noise_tolerance.npz")
    if not path.exists():
        return None
    data = np.load(path)
    return data["noise_levels"] * 100.0, data["relative_rms_mean"]


def make_figure(inversion_domain, results, images, truth, output_path):
    levels = np.array(CALIBRATION_LEVELS) * 100.0
    means = np.array([np.mean(results[level]) for level in CALIBRATION_LEVELS])
    lows = np.array([np.min(results[level]) for level in CALIBRATION_LEVELS])
    highs = np.array([np.max(results[level]) for level in CALIBRATION_LEVELS])

    fig, axes = plt.subplots(1, 5, figsize=(23, 4.4))
    axes[0].fill_between(levels, lows, highs, alpha=0.25, color="darkorange",
                         label="min-max over devices")
    axes[0].plot(levels, means, "o-", color="darkorange", lw=1.8, label="calibration error")
    noise = load_noise_curve()
    if noise is not None:
        axes[0].plot(noise[0], noise[1], "s--", color="steelblue", lw=1.4,
                     label="random noise (step 06)")
    axes[0].axhline(1.0, color="crimson", ls=":", label="no recovery")
    axes[0].set_title("Calibration vs noise tolerance")
    axes[0].set_xlabel("per-element error level (% amplitude / rad phase)")
    axes[0].set_ylabel("relative RMS error")
    axes[0].set_ylim(0, 1.25)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    x = inversion_domain.coordinates[:, 0]
    y = inversion_domain.coordinates[:, 1]
    tri = inversion_domain.triangles
    panels = [(axes[1], truth, "Truth c(x)  [m/s]")]
    for offset, level in enumerate(IMAGE_LEVELS):
        panels.append((axes[2 + offset], images[level],
                       f"Recovered @ {level*100:.0f}% calib  [m/s]"))
    for ax, values, title in panels:
        tpc = ax.tricontourf(x, y, tri, values, levels=24, cmap="inferno", vmin=1460, vmax=1560)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Calibration-tolerance curve: systematic per-element gain/phase error "
        "(coherent -- unlike random noise)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    observation_domain = build_disk(RADIUS, refinements=7)
    inversion_domain = build_disk(RADIUS, refinements=6)
    print("=== Calibration-tolerance sweep (radiating boundary, SI) ===")
    print(f"observation mesh {observation_domain.basis.N} dofs "
          f"({observation_domain.boundary_dofs.size} boundary elements), "
          f"inversion mesh {inversion_domain.basis.N} dofs; {len(SEEDS)} devices per level")

    results, images, truth = run_sweep(observation_domain, inversion_domain)

    knee = next((lv for lv in CALIBRATION_LEVELS if np.mean(results[lv]) > 0.5), None)
    print("\n=== Summary ===")
    print(f"clean relative RMS: {np.mean(results[0.0]):.3f}")
    if knee is not None:
        print(f"breakdown (mean relative RMS > 0.5) first at {knee*100:.0f}% "
              f"amplitude / {np.degrees(knee):.1f} deg phase")
    else:
        print("no breakdown up to the maximum swept calibration error")
    noise = load_noise_curve()
    if noise is not None:
        noise_knee = next((lv for lv, r in zip(noise[0], noise[1], strict=True) if r > 0.5), None)
        print(f"(for contrast, random noise broke near "
              f"{noise_knee:.0f}% -- calibration is the harder, coherent error)")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    figure_path = output_dir / "calibration_tolerance.png"
    make_figure(inversion_domain, results, images, truth, figure_path)
    np.savez_compressed(
        output_dir / "calibration_tolerance.npz",
        levels=np.array(CALIBRATION_LEVELS),
        relative_rms_mean=np.array([np.mean(results[lv]) for lv in CALIBRATION_LEVELS]),
    )
    print(f"\nfigure -> {figure_path}")


if __name__ == "__main__":
    main()

"""Realistic-contrast reconstruction in SI units on the radiating boundary.

The honest translation test. Everything here is in real units -- metres, metres
per second, hertz -- and the contrasts are the few-percent values real soft tissue
actually shows (harder than the 8-15% toy contrasts used to prove the algorithm).
It runs the physically faithful radiating-boundary forward, frequency
continuation, data on a disjoint fine mesh with measurement noise, and reports
everything back in SI (m/s, kHz, points per wavelength).

Setup (a small water-tank / breast-scale disk):
  * radius 0.10 m, water bath c = 1500 m/s (known, frozen at the boundary),
  * gland (+1.7%), tumour (+3.7%), cyst (-2.3%) -- max contrast under 4%,
  * frequency continuation at kR = 5, 10, 16  (f = 11.9, 23.9, 38.2 kHz).

Run:  python research/si_reconstruction.py
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

RADIUS = 0.10  # metres
BATH_SPEED = 1500.0  # m/s (water)
NOISE_FRACTION = 0.01
SPEED_BOUNDS = (1400.0, 1600.0)  # m/s, realistic soft-tissue range

# Realistic few-percent tissue contrasts (absolute m/s added to the water bath).
TRUTH_FEATURES = (
    {"type": "ellipse", "label": "gland", "center": (0.0, 0.0),
     "semi_axes": (0.045, 0.035), "contrast": 25.0},   # 1525 m/s, +1.7%
    {"type": "disk", "label": "tumour", "center": (0.035, -0.02),
     "radius": 0.020, "contrast": 55.0},                # 1555 m/s, +3.7%
    {"type": "disk", "label": "cyst", "center": (-0.03, 0.03),
     "radius": 0.018, "contrast": -35.0},               # 1465 m/s, -2.3%
)
TRUTH_SHARPNESS = 250.0  # 1/m; ~4 mm edges

K_RADII = (5.0, 10.0, 16.0)
# H1 smoothness weights in the NORMALIZED variable u = c / bath (order 1). The
# forward/adjoint run in SI; only the optimizer's search variable is normalized.
# Kept small so the prior only suppresses noise-scale roughness rather than the
# (already weak) tissue contrast itself.
ALPHAS = (3e-5, 1e-5, 3e-6)
MAX_ITERATIONS = 60


def frequency_for(k_radius: float) -> float:
    """SI temporal frequency (Hz) giving this kR at the bath speed on this disk."""

    return k_radius * BATH_SPEED / (2.0 * pi * RADIUS)


def patterns_for(k_radius: float) -> tuple[str, ...]:
    max_mode = max(2, ceil(k_radius))
    modes = [pattern for m in range(1, max_mode + 1) for pattern in (f"cos:{m}", f"sin:{m}")]
    return tuple(["constant"] + modes)


def truth_speed(domain):
    return feature_speed(domain, BATH_SPEED, TRUTH_SHARPNESS, TRUTH_FEATURES).sound_speed


def points_per_wavelength(domain, k_radius):
    wavelength = 2.0 * pi * RADIUS / k_radius
    return wavelength / domain.h_max


def si_invariance_check(observation_domain, refinements):
    """Print the SI/dimensionless forward ratio -- must be the real constant RADIUS."""

    unit_domain = build_disk(1.0, refinements)
    x = unit_domain.coordinates[:, 0]
    y = unit_domain.coordinates[:, 1]
    contrast = 0.02 * np.exp(-((x - 0.2) ** 2 + (y + 0.1) ** 2) / 0.1)
    forcing_unit = build_boundary_forcing(unit_domain, ("cos:1", "sin:2"))
    pressure_unit = forward_boundary_pressure(
        unit_domain, 1.0 + contrast, forcing_unit, 5.0, 0.0, 1.0
    )

    forcing_si = build_boundary_forcing(observation_domain, ("cos:1", "sin:2"))
    si_speed = BATH_SPEED * (1.0 + contrast)
    pressure_si = forward_boundary_pressure(
        observation_domain, si_speed, forcing_si, 5.0 * BATH_SPEED / RADIUS, 0.0, BATH_SPEED
    )
    ratio = pressure_si / (RADIUS * pressure_unit)
    return float(np.max(np.abs(ratio - 1.0)))


def synthesize(observation_domain, inversion_domain, rng):
    truth = truth_speed(observation_domain)
    observed = {}
    for index, k_radius in enumerate(K_RADII):
        forcing = build_boundary_forcing(observation_domain, patterns_for(k_radius))
        omega = 2.0 * pi * frequency_for(k_radius)
        clean = forward_boundary_pressure(
            observation_domain, truth, forcing, omega, 0.0, BATH_SPEED
        )
        rms = np.sqrt(np.mean(np.abs(clean) ** 2))
        noise = rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
        noisy = clean + NOISE_FRACTION * rms / np.sqrt(2.0) * noise
        observed[index] = resample_flux_to(
            inversion_domain.boundary_theta, observation_domain.boundary_theta, noisy
        )
    return observed


def reconstruct(inversion_domain, observed):
    """Frequency-continuation FWI in SI, searching the normalized variable u = c/bath.

    The radiating forward and its adjoint run entirely in SI units (metres, m/s,
    Hz); the optimizer variable is normalized so the objective is well-scaled and
    the H1 prior uses ordinary order-1 weights. Data gradient in SI is chained to
    the normalized variable by dJ/du = bath * dJ/dc.
    """

    boundary_mass_full = full_boundary_mass(inversion_domain)
    boundary_mass = boundary_mass_matrix(inversion_domain)
    stiffness = asm(_stiffness_form, inversion_domain.basis).tocsc()
    boundary = inversion_domain.boundary_dofs

    normalized = np.ones(inversion_domain.basis.N)  # u = c / bath
    low, high = SPEED_BOUNDS[0] / BATH_SPEED, SPEED_BOUNDS[1] / BATH_SPEED
    bounds = [(low, high)] * inversion_domain.basis.N
    for dof in boundary:
        bounds[int(dof)] = (1.0, 1.0)  # bath frozen at the boundary

    history: list[float] = []
    reports = []
    for index, k_radius in enumerate(K_RADII):
        forcing = build_boundary_forcing(inversion_domain, patterns_for(k_radius))
        omega = 2.0 * pi * frequency_for(k_radius)
        observed_flux = observed[index]
        alpha = ALPHAS[index]

        def evaluate(candidate, _forcing=forcing, _omega=omega,
                     _observed=observed_flux, _alpha=alpha):
            speed = candidate * BATH_SPEED  # SI forward/adjoint
            result = radiating_objective_and_gradient(
                inversion_domain, speed, _forcing, _omega, 0.0, BATH_SPEED, _observed,
                alpha=0.0, background_speed=BATH_SPEED,
                boundary_mass_full=boundary_mass_full, boundary_mass=boundary_mass,
            )
            gradient = result.gradient * BATH_SPEED  # dJ/du = bath * dJ/dc
            # H1 smoothness prior in the normalized variable.
            stiffness_u = np.asarray(stiffness @ candidate, dtype=np.float64)
            reg_value = 0.5 * _alpha * float(candidate @ stiffness_u)
            gradient = gradient + _alpha * stiffness_u
            gradient[boundary] = 0.0
            history.append(result.data_misfit)
            return result.data_misfit + reg_value, gradient

        initial_misfit, _ = evaluate(normalized)
        optimized = minimize(evaluate, normalized, jac=True, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": MAX_ITERATIONS, "ftol": 1e-14, "gtol": 1e-12})
        normalized = np.clip(optimized.x, low, high)
        final, _ = evaluate(normalized)
        reports.append((k_radius, frequency_for(k_radius), len(forcing.names),
                        points_per_wavelength(inversion_domain, k_radius), initial_misfit, final))
    return normalized * BATH_SPEED, history, reports


def make_figure(inversion_domain, recovered, history, output_path):
    truth = truth_speed(inversion_domain)
    x = inversion_domain.coordinates[:, 0]
    y = inversion_domain.coordinates[:, 1]
    tri = inversion_domain.triangles
    error = recovered - truth

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for ax, values, title, cmap, limits in (
        (axes[0], truth, "Truth c(x)  [m/s]", "inferno", (1460, 1560)),
        (axes[1], recovered, "Recovered c(x)  [m/s]", "inferno", (1460, 1560)),
        (axes[2], error, "Error  [m/s]", "RdBu_r", (-30, 30)),
    ):
        tpc = ax.tricontourf(x, y, tri, values, levels=24, cmap=cmap,
                             vmin=limits[0], vmax=limits[1])
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    axes[3].semilogy(np.arange(len(history)), history, lw=1.5)
    axes[3].set_title("Data misfit (all stages)")
    axes[3].set_xlabel("objective evaluation")
    axes[3].set_ylabel("misfit")
    axes[3].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Realistic-contrast SI reconstruction (radiating boundary): "
        "0.10 m disk, water 1500 m/s, tissue contrasts under 4%, 1% noise",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    rng = np.random.default_rng(20260727)
    observation_domain = build_disk(RADIUS, refinements=7)
    inversion_domain = build_disk(RADIUS, refinements=6)

    print("=== Realistic-contrast SI reconstruction (radiating boundary) ===")
    print(f"disk radius {RADIUS} m, water bath {BATH_SPEED} m/s")
    print(f"observation mesh {observation_domain.basis.N} dofs, inversion mesh "
          f"{inversion_domain.basis.N} dofs (disjoint)")
    true_full = truth_speed(inversion_domain)
    print(f"true tissue speeds: [{true_full.min():.1f}, {true_full.max():.1f}] m/s  "
          f"(contrast {(true_full.min()/BATH_SPEED-1)*100:+.1f}% .. "
          f"{(true_full.max()/BATH_SPEED-1)*100:+.1f}%)")

    invariance = si_invariance_check(observation_domain, 7)
    print(f"SI/dimensionless forward agreement: max deviation {invariance:.2e} "
          f"(0 == units are exact)")

    for k_radius in K_RADII:
        print(f"  stage kR={k_radius:.0f}  f={frequency_for(k_radius)/1e3:.1f} kHz  "
              f"modes={len(patterns_for(k_radius))}  "
              f"pts/wavelength={points_per_wavelength(inversion_domain, k_radius):.1f}")

    observed = synthesize(observation_domain, inversion_domain, rng)
    recovered, history, reports = reconstruct(inversion_domain, observed)

    print("\n=== Per-stage data misfit ===")
    for k_radius, frequency, modes, ppw, initial, final in reports:
        print(f"  kR={k_radius:.0f} ({frequency/1e3:.1f} kHz, {modes} modes, "
              f"{ppw:.0f} pts/wavelength): {initial:.3e} -> {final:.3e}")

    truth = truth_speed(inversion_domain)
    interior = inversion_domain.interior_dofs
    error = recovered[interior] - truth[interior]
    rms_error = float(np.sqrt(np.mean(error**2)))
    baseline = float(np.sqrt(np.mean((truth[interior] - BATH_SPEED) ** 2)))
    print("\n=== Recovery metrics (SI) ===")
    print(f"recovered speeds: [{recovered.min():.1f}, {recovered.max():.1f}] m/s")
    print(f"RMS error: {rms_error:.2f} m/s ({rms_error/BATH_SPEED*100:.2f}% of bath speed); "
          f"relative RMS {rms_error/baseline:.3f}")
    print(f"peak contrast recovered: {(recovered.max()/BATH_SPEED-1)*100:+.2f}% "
          f"(true tumour {(truth.max()/BATH_SPEED-1)*100:+.2f}%)")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    figure_path = output_dir / "si_reconstruction.png"
    make_figure(inversion_domain, recovered, history, figure_path)
    print(f"\nfigure -> {figure_path}")


if __name__ == "__main__":
    main()

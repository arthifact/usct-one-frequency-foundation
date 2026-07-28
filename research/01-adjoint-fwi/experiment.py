"""End-to-end hypothesis test: reconstruct c(x) from boundary-flux data.

This is the research payload experiment. It exercises the whole loop the project
was built toward:

    known truth phantom
      -> synthesize complex boundary flux on a FINE observation mesh
      -> add measurement noise
      -> resample onto a COARSER, disjoint inversion mesh (no inverse crime)
      -> frequency-continuation complex-L2 FWI (adjoint gradient, L-BFGS-B)
      -> recovered c(x)

It reports quantitative recovery metrics and writes a figure. It is deliberately
dimensionless (unit disk, water c = 1) to match the verified foundation; SI
scaling and the kR wall are a separate study.

Run:  python research/fwi_experiment.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from usct.dtn.inversion import forward_flux  # noqa: E402
from usct.dtn.reconstruction import Stage, adaptive_patterns, reconstruct  # noqa: E402
from usct.physics.boundary import resample_flux_to  # noqa: E402
from usct.physics.domain import build_disk  # noqa: E402
from usct.physics.forcing import build_boundary_forcing  # noqa: E402
from usct.physics.medium import feature_speed  # noqa: E402

RADIUS = 1.0
LOSS = 1e-3  # small stabilizer against interior Dirichlet resonances (known constant)
NOISE_FRACTION = 0.01  # complex Gaussian noise at 1% of per-stage RMS flux
SPEED_BOUNDS = (0.85, 1.30)
BACKGROUND = 1.0  # immersion medium (water) known -> boundary speed frozen

TRUTH_FEATURES = (
    {"type": "disk", "label": "tumor", "center": (0.22, -0.16), "radius": 0.30, "contrast": 0.15},
    {"type": "disk", "label": "cyst", "center": (-0.25, 0.22), "radius": 0.16, "contrast": -0.08},
)
TRUTH_SHARPNESS = 20.0

# Low -> high frequency continuation. kR = 2 pi f at c = 1.
FREQUENCIES = (3.0 / (2 * np.pi), 5.0 / (2 * np.pi), 8.0 / (2 * np.pi))
ALPHAS = (3e-4, 1e-4, 3e-5)
MAX_ITERATIONS = (40, 40, 40)


def truth_speed(domain):
    return feature_speed(domain, BACKGROUND, TRUTH_SHARPNESS, TRUTH_FEATURES).sound_speed


def synthesize_observations(observation_domain, inversion_domain, stages, rng):
    """Forward-model each stage on the fine mesh, add noise, resample to coarse mesh."""

    truth = truth_speed(observation_domain)
    observed: dict[int, np.ndarray] = {}
    for index, stage in enumerate(stages):
        forcing = build_boundary_forcing(observation_domain, stage.patterns)
        omega = 2.0 * np.pi * stage.frequency
        clean = forward_flux(observation_domain, truth, forcing, omega, stage.loss).flux
        rms = np.sqrt(np.mean(np.abs(clean) ** 2))
        noise = (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
        noisy = clean + NOISE_FRACTION * rms / np.sqrt(2.0) * noise
        observed[index] = resample_flux_to(
            inversion_domain.boundary_theta, observation_domain.boundary_theta, noisy
        )
    return observed


def cold_start_highest(inversion_domain, observed, stages):
    """Control: invert ONLY the highest frequency from a homogeneous guess.

    Frequency continuation exists to dodge cycle-skipping. Skipping it -- starting
    the highest frequency cold -- is the canonical way to fall into a local
    minimum. Given the same total iteration budget, this shows whether the
    low-to-high schedule actually earned its keep at this contrast.
    """

    last = len(stages) - 1
    cold_stage = Stage(
        frequency=stages[last].frequency,
        patterns=stages[last].patterns,
        loss=stages[last].loss,
        alpha=stages[last].alpha,
        max_iterations=sum(stage.max_iterations for stage in stages),
    )
    initial = np.full(inversion_domain.basis.N, BACKGROUND, dtype=np.float64)
    result = reconstruct(
        inversion_domain, {0: observed[last]}, (cold_stage,),
        initial_speed=initial, speed_bounds=SPEED_BOUNDS, background_speed=BACKGROUND,
    )
    return result.sound_speed


def recovery_metrics(inversion_domain, recovered):
    truth = truth_speed(inversion_domain)
    interior = inversion_domain.interior_dofs
    error = recovered[interior] - truth[interior]
    rms_error = float(np.sqrt(np.mean(error**2)))
    baseline = float(np.sqrt(np.mean((truth[interior] - BACKGROUND) ** 2)))
    return {
        "rms_error": rms_error,
        "rms_error_vs_homogeneous": float(np.sqrt(np.mean((BACKGROUND - truth[interior]) ** 2))),
        "relative_rms_error": rms_error / baseline,
        "true_speed_range": (float(truth.min()), float(truth.max())),
        "recovered_speed_range": (float(recovered.min()), float(recovered.max())),
    }


def make_figure(inversion_domain, recovered, cold_recovered, result, output_path):
    truth = truth_speed(inversion_domain)
    x = inversion_domain.coordinates[:, 0]
    y = inversion_domain.coordinates[:, 1]
    triangles = inversion_domain.triangles
    vmin, vmax = SPEED_BOUNDS

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    for ax, values, title, cmap, limits in (
        (axes[0], truth, "Truth c(x)", "inferno", (vmin, vmax)),
        (axes[1], recovered, "Recovered: continuation", "inferno", (vmin, vmax)),
        (axes[2], cold_recovered, "Recovered: cold-start high f", "inferno", (vmin, vmax)),
        (axes[3], recovered - truth, "Error (continuation)", "RdBu_r", (-0.1, 0.1)),
    ):
        tpc = ax.tricontourf(x, y, triangles, values, levels=24, cmap=cmap,
                             vmin=limits[0], vmax=limits[1])
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    axes[4].semilogy(np.arange(len(result.misfit_history)), result.misfit_history, lw=1.5)
    axes[4].set_title("Data misfit (all stages)")
    axes[4].set_xlabel("objective evaluation")
    axes[4].set_ylabel("misfit")
    axes[4].grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Frequency-continuation FWI: sound-speed recovery from boundary flux "
        "(data on a disjoint fine mesh + 1% noise)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    rng = np.random.default_rng(20260727)
    observation_domain = build_disk(RADIUS, refinements=7)
    inversion_domain = build_disk(RADIUS, refinements=5)

    stages = tuple(
        Stage(
            frequency=frequency,
            patterns=adaptive_patterns(frequency, RADIUS),
            loss=LOSS,
            alpha=alpha,
            max_iterations=iterations,
        )
        for frequency, alpha, iterations in zip(FREQUENCIES, ALPHAS, MAX_ITERATIONS, strict=True)
    )

    print("=== Frequency-continuation FWI hypothesis test ===")
    print(f"observation mesh: {observation_domain.basis.N} dofs (refinements=7)")
    print(f"inversion mesh:   {inversion_domain.basis.N} dofs (refinements=5)")
    for stage in stages:
        print(f"  stage f={stage.frequency:.4f}  kR~{2*np.pi*stage.frequency:.1f}  "
              f"modes={len(stage.patterns)}  alpha={stage.alpha:.1e}")

    observed = synthesize_observations(observation_domain, inversion_domain, stages, rng)

    initial = np.full(inversion_domain.basis.N, BACKGROUND, dtype=np.float64)
    result = reconstruct(
        inversion_domain, observed, stages,
        initial_speed=initial, speed_bounds=SPEED_BOUNDS,
        background_speed=BACKGROUND, progress=print,
    )

    metrics = recovery_metrics(inversion_domain, result.sound_speed)
    print("\n=== Recovery metrics (frequency continuation) ===")
    print(f"true speed range:      [{metrics['true_speed_range'][0]:.4f}, "
          f"{metrics['true_speed_range'][1]:.4f}]")
    print(f"recovered speed range: [{metrics['recovered_speed_range'][0]:.4f}, "
          f"{metrics['recovered_speed_range'][1]:.4f}]")
    print(f"RMS error (interior):  {metrics['rms_error']:.5f}")
    print(f"  vs homogeneous guess: {metrics['rms_error_vs_homogeneous']:.5f} "
          f"(relative RMS error {metrics['relative_rms_error']:.3f})")

    print("\n=== Control: highest frequency cold-started (no continuation) ===")
    cold_recovered = cold_start_highest(inversion_domain, observed, stages)
    cold_metrics = recovery_metrics(inversion_domain, cold_recovered)
    print(f"recovered speed range: [{cold_metrics['recovered_speed_range'][0]:.4f}, "
          f"{cold_metrics['recovered_speed_range'][1]:.4f}]")
    print(f"RMS error (interior):  {cold_metrics['rms_error']:.5f} "
          f"(relative RMS error {cold_metrics['relative_rms_error']:.3f})")
    print(f"continuation improves RMS by "
          f"{(1 - metrics['rms_error'] / cold_metrics['rms_error']) * 100:.1f}% vs cold start")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    figure_path = output_dir / "fwi_reconstruction.png"
    make_figure(inversion_domain, result.sound_speed, cold_recovered, result, figure_path)
    np.savez_compressed(
        output_dir / "fwi_reconstruction.npz",
        recovered=result.sound_speed,
        truth=truth_speed(inversion_domain),
        coordinates=inversion_domain.coordinates,
        triangles=inversion_domain.triangles,
        misfit_history=np.asarray(result.misfit_history),
    )
    print(f"\nfigure -> {figure_path}")
    print(f"data   -> {output_dir / 'fwi_reconstruction.npz'}")


if __name__ == "__main__":
    main()

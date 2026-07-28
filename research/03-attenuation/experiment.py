"""Physics fidelity: attenuation damps resonances, and can be imaged.

Two demonstrations built on the verified joint (c, eta) adjoint
(`src/usct/dtn/attenuation.py`):

1. RESONANCE DAMPING. The lossless hard-Dirichlet disk is a resonant cavity: the
   boundary response blows up whenever kR hits an interior Dirichlet eigenvalue
   (a Bessel zero). A physical, spatially-distributed loss moves those
   eigenvalues off the real axis and bounds the response -- turning the numerical
   `eta` stabilizer into an honest attenuation model.

2. ATTENUATION IMAGING. With sound speed known, reconstruct a spatially-varying
   attenuation field eta(x) from boundary flux (data on a disjoint mesh + noise).
   Absorption has a weaker, more diffuse boundary signature than speed, so this
   is a genuinely harder inverse problem -- an honest look at what is recoverable.

Run:  python research/attenuation_experiment.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from usct.dtn.attenuation import forward_flux_complex, joint_objective_and_gradient  # noqa: E402
from usct.physics.boundary import (
    boundary_mass_matrix,  # noqa: E402
    resample_flux_to,  # noqa: E402
)
from usct.physics.domain import build_disk  # noqa: E402
from usct.physics.forcing import build_boundary_forcing  # noqa: E402
from usct.physics.medium import feature_speed  # noqa: E402


def resonance_sweep():
    """Max boundary-flux amplitude vs kR for a fixed mode, lossless vs attenuating."""

    domain = build_disk(1.0, refinements=6)
    forcing = build_boundary_forcing(domain, ("exp:2",))  # resonances at j_{2,l}
    speed = np.ones(domain.basis.N)
    k_values = np.linspace(3.0, 12.0, 140)
    losses = (0.0, 0.005, 0.02)
    curves = {loss: [] for loss in losses}
    for k_radius in k_values:
        for loss in losses:
            flux = forward_flux_complex(domain, speed, loss, forcing, k_radius)
            curves[loss].append(float(np.max(np.abs(flux))))
    return k_values, {loss: np.asarray(values) for loss, values in curves.items()}


def attenuation_reconstruction():
    """Recover an absorption inclusion eta(x) at known sound speed."""

    observation_domain = build_disk(1.0, refinements=6)
    inversion_domain = build_disk(1.0, refinements=5)
    rng = np.random.default_rng(7)

    # Truth: homogeneous water speed, one lossy inclusion on a low-loss background.
    background_loss = 0.005
    loss_features = (
        {"type": "disk", "label": "absorber", "center": (-0.1, 0.15),
         "radius": 0.26, "contrast": 0.06},
    )
    truth_loss = feature_speed(observation_domain, background_loss, 18.0, loss_features).sound_speed
    speed_obs = np.ones(observation_domain.basis.N)

    frequency = 8.0 / (2 * np.pi)
    omega = 2.0 * np.pi * frequency
    patterns = tuple(["constant"] + [p for m in range(1, 9) for p in (f"cos:{m}", f"sin:{m}")])
    forcing_obs = build_boundary_forcing(observation_domain, patterns)
    clean = forward_flux_complex(observation_domain, speed_obs, truth_loss, forcing_obs, omega)
    rms = np.sqrt(np.mean(np.abs(clean) ** 2))
    noise = rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    noisy = clean + 0.01 * rms / np.sqrt(2.0) * noise
    observed = resample_flux_to(
        inversion_domain.boundary_theta, observation_domain.boundary_theta, noisy
    )

    forcing_inv = build_boundary_forcing(inversion_domain, patterns)
    speed_known = np.ones(inversion_domain.basis.N)
    boundary_mass = boundary_mass_matrix(inversion_domain)
    boundary = inversion_domain.boundary_dofs
    history: list[float] = []

    def objective(eta_vector):
        eta = eta_vector.copy()
        eta[boundary] = background_loss
        result = joint_objective_and_gradient(
            inversion_domain, speed_known, eta, forcing_inv, omega, observed,
            alpha_loss=2e-4, background_loss=background_loss, boundary_mass=boundary_mass,
        )
        history.append(result.data_misfit)
        return result.misfit, result.loss_gradient

    initial = np.full(inversion_domain.basis.N, background_loss)
    bounds = [(0.0, 0.15)] * inversion_domain.basis.N
    for dof in boundary:
        bounds[int(dof)] = (background_loss, background_loss)
    optimized = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 80, "ftol": 1e-12, "gtol": 1e-9})

    recovered = optimized.x
    truth_inv = feature_speed(inversion_domain, background_loss, 18.0, loss_features).sound_speed
    interior = inversion_domain.interior_dofs
    error = recovered[interior] - truth_inv[interior]
    rms_error = float(np.sqrt(np.mean(error**2)))
    baseline = float(np.sqrt(np.mean((truth_inv[interior] - background_loss) ** 2)))
    return {
        "domain": inversion_domain,
        "truth": truth_inv,
        "recovered": recovered,
        "history": history,
        "rms_error": rms_error,
        "relative_rms_error": rms_error / baseline,
        "true_range": (float(truth_inv.min()), float(truth_inv.max())),
        "recovered_range": (float(recovered.min()), float(recovered.max())),
    }


def make_figure(k_values, curves, recon, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

    for loss, values in curves.items():
        label = "lossless (eta=0)" if loss == 0.0 else f"eta={loss}"
        axes[0].semilogy(k_values, values, lw=1.6, label=label)
    axes[0].set_title("Resonance damping (mode m=2)")
    axes[0].set_xlabel("kR")
    axes[0].set_ylabel("max |boundary flux|")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    domain = recon["domain"]
    x, y, tri = domain.coordinates[:, 0], domain.coordinates[:, 1], domain.triangles
    vmax = max(recon["true_range"][1], recon["recovered_range"][1])
    for ax, values, title in (
        (axes[1], recon["truth"], "Truth attenuation eta(x)"),
        (axes[2], recon["recovered"], "Recovered eta(x)"),
    ):
        tpc = ax.tricontourf(x, y, tri, values, levels=24, cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    axes[3].semilogy(np.arange(len(recon["history"])), recon["history"], lw=1.5)
    axes[3].set_title("Attenuation-inversion misfit")
    axes[3].set_xlabel("objective evaluation")
    axes[3].set_ylabel("data misfit")
    axes[3].grid(True, which="both", alpha=0.3)

    fig.suptitle("Physics fidelity: attenuation damps resonances (left) and is imageable (right)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    print("=== Resonance damping sweep (mode m=2, kR in [3, 12]) ===")
    k_values, curves = resonance_sweep()
    peak_lossless = float(np.max(curves[0.0]))
    peak_damped = float(np.max(curves[0.02]))
    print(f"peak max|flux|: lossless = {peak_lossless:.2e},  eta=0.02 = {peak_damped:.2e}  "
          f"({peak_lossless / peak_damped:.1f}x reduction at the worst resonance)")

    print("\n=== Attenuation imaging (sound speed known) ===")
    recon = attenuation_reconstruction()
    print(f"true eta range:      [{recon['true_range'][0]:.4f}, {recon['true_range'][1]:.4f}]")
    print(f"recovered eta range: [{recon['recovered_range'][0]:.4f}, "
          f"{recon['recovered_range'][1]:.4f}]")
    print(f"RMS error: {recon['rms_error']:.5f}  (relative {recon['relative_rms_error']:.3f})")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    figure_path = output_dir / "attenuation.png"
    make_figure(k_values, curves, recon, figure_path)
    print(f"\nfigure -> {figure_path}")


if __name__ == "__main__":
    main()

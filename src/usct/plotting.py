"""Noninteractive plots of saved one-frequency forward results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from usct.results import SavedResult, load_result

_GUIDE_COLOR = "#176B87"


def _pattern_index(saved: SavedResult, pattern_name: str | None) -> int:
    if pattern_name is None:
        return 0
    try:
        return saved.forcing_names.index(pattern_name)
    except ValueError as error:
        choices = ", ".join(saved.forcing_names)
        message = f"unknown pattern {pattern_name!r}; available patterns: {choices}"
        raise ValueError(message) from error


def _explained_title(axis, title, explanation):
    axis.set_title(title, pad=19)
    axis.text(
        0.5,
        1.015,
        explanation,
        color=_GUIDE_COLOR,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="bottom",
        transform=axis.transAxes,
    )


def _field_panel(figure, axis, triangulation, values, title, explanation, cmap="viridis"):
    artist = axis.tripcolor(triangulation, values, shading="gouraud", cmap=cmap)
    _explained_title(axis, title, explanation)
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.colorbar(artist, ax=axis, shrink=0.78)


def plot_saved_result(
    saved: SavedResult,
    output_path: Path,
    pattern_name: str | None = None,
) -> Path:
    """Create the required seven-view overview PNG from saved arrays only."""

    index = _pattern_index(saved, pattern_name)
    name = saved.forcing_names[index]
    field = saved.fields[:, index]
    triangulation = Triangulation(
        saved.coordinates[:, 0],
        saved.coordinates[:, 1],
        saved.triangles,
    )
    figure, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    _field_panel(
        figure,
        axes[0, 0],
        triangulation,
        saved.sound_speed,
        "Sound speed c(x)",
        "material map",
    )
    _field_panel(
        figure,
        axes[0, 1],
        triangulation,
        field.real,
        f"Re(u): {name}",
        "in-phase pressure",
        "RdBu_r",
    )
    _field_panel(
        figure,
        axes[0, 2],
        triangulation,
        field.imag,
        f"Im(u): {name}",
        "90°-shifted pressure",
        "RdBu_r",
    )
    _field_panel(
        figure,
        axes[0, 3],
        triangulation,
        np.abs(field),
        f"|u|: {name}",
        "pressure strength",
    )
    _field_panel(
        figure,
        axes[1, 0],
        triangulation,
        np.angle(field),
        f"phase(u): {name}",
        "wave timing [rad]",
        "twilight",
    )
    theta = saved.boundary_theta
    forcing = saved.forcing_values[:, index]
    response = saved.boundary_flux_response[:, index]
    axes[1, 1].plot(theta, forcing.real, label="I = real")
    axes[1, 1].plot(theta, forcing.imag, label="Q = imag")
    _explained_title(axes[1, 1], "Prescribed boundary pressure", "input we drive")
    axes[1, 2].plot(theta, response.real, label="I = real")
    axes[1, 2].plot(theta, response.imag, label="Q = imag")
    _explained_title(axes[1, 2], "Observed outward boundary flux", "output we measure")
    for axis in axes[1, 1:3]:
        axis.set_xlabel("boundary angle θ [rad]")
        axis.grid(alpha=0.25)
        axis.legend()
    guide_lines = (
        "QUICK KEY\nRe = in-phase    Im = 90-degree shifted\n|u| = strength  phase = wave timing"
    )
    metadata_lines = (
        "Idealized Dirichlet-to-Neumann response\n"
        f"f = {saved.metadata['frequency']:.6g}\n"
        f"omega = {saved.metadata['angular_frequency']:.6g}\n"
        f"loss eta = {saved.metadata['loss']:.6g}\n"
        f"dofs = {saved.metadata['degrees_of_freedom']}\n"
        f"points/wavelength = {saved.metadata['points_per_wavelength']:.2f}\n"
        f"time convention: {saved.metadata['time_convention']}"
    )
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.02,
        0.95,
        guide_lines,
        color=_GUIDE_COLOR,
        fontweight="bold",
        va="top",
        family="monospace",
    )
    axes[1, 3].text(0.02, 0.72, metadata_lines, va="top", family="monospace")
    figure.suptitle("USCT one-frequency forward simulation")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def plot_result(
    output_dir: Path,
    pattern_name: str | None = None,
) -> Path:
    """Load an existing result and create ``overview.png`` without simulating."""

    output = Path(output_dir)
    return plot_saved_result(load_result(output), output / "overview.png", pattern_name)

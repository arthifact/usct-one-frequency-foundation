"""Turn a solved pipeline result into displayable grids and boundary traces.

Shared by both front-ends (the local web studio and the native Qt studio) so
they rasterize the FEM fields identically. Everything here is pure NumPy/SciPy;
no UI imports.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import griddata

from usct.engine.pipeline import PipelineResult


def domain_radius(coordinates: NDArray[np.float64]) -> float:
    """Largest node distance from the origin (the disk radius)."""

    return float(np.max(np.hypot(coordinates[:, 0], coordinates[:, 1])))


def field_values(
    result: PipelineResult, view: str, pattern: int
) -> tuple[NDArray[np.float64], int, int]:
    """Return the nodal values to display, the clamped pattern, and pattern count."""

    speed = result.medium.sound_speed
    field = result.forward.solution.field
    count = field.shape[1]
    pattern = max(0, min(pattern, count - 1))
    values = field[:, pattern].real if view == "wavefield" else speed
    return np.asarray(values, dtype=np.float64), pattern, count


def rasterize(
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64],
    radius: float,
    n: int,
) -> NDArray[np.float64]:
    """Interpolate nodal ``values`` onto an ``n x n`` grid, ``NaN`` outside the disk.

    Row 0 is the bottom edge (``y = -radius``); pair with ``origin="lower"`` and
    ``extent=[-radius, radius, -radius, radius]`` for physically upright display.
    """

    axis = np.linspace(-radius, radius, n)
    gx, gy = np.meshgrid(axis, axis)
    grid = griddata(coordinates, values, (gx, gy), method="linear")
    grid[(gx**2 + gy**2) > radius**2] = np.nan
    return np.asarray(grid, dtype=np.float64)


def boundary_iq(
    result: PipelineResult, pattern: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return angle-sorted ``(theta, I, Q)`` for one forcing pattern."""

    response = result.response
    order = np.argsort(response.theta)
    theta = response.theta[order]
    channel = response.complex_response[order, pattern]
    return theta, np.asarray(channel.real), np.asarray(channel.imag)

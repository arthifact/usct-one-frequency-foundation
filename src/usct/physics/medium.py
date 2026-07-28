"""Transparent deterministic sound-speed models."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usct.physics.domain import Domain


@dataclass(frozen=True)
class Medium:
    """Nodal sound speed with shape ``(number_of_dofs,)``."""

    sound_speed: NDArray[np.float64]
    description: str


def _checked_medium(domain: Domain, values: NDArray[np.float64], description: str) -> Medium:
    sound_speed = np.asarray(values, dtype=np.float64)
    expected = (domain.basis.N,)
    if sound_speed.shape != expected:
        raise ValueError(f"sound-speed shape must be {expected}; got {sound_speed.shape}")
    if not np.all(np.isfinite(sound_speed)):
        raise ValueError("sound speed must contain only finite values")
    if np.any(sound_speed <= 0.0):
        raise ValueError("sound speed must be strictly positive")
    return Medium(sound_speed=sound_speed.copy(), description=description)


def constant_speed(domain: Domain, speed: float) -> Medium:
    """Return a spatially constant sound speed in configured speed units."""

    values = np.full(domain.basis.N, speed, dtype=np.float64)
    return _checked_medium(domain, values, f"constant sound speed c={speed:g}")


def circular_inclusion(
    domain: Domain,
    background_speed: float,
    center: tuple[float, float],
    radius: float,
    inclusion_speed: float,
) -> Medium:
    """Return a nodal two-speed disk with one circular demonstration inclusion."""

    if len(center) != 2 or not np.all(np.isfinite(center)):
        raise ValueError("inclusion center must contain two finite coordinates")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("inclusion radius must be positive and finite")
    coordinates = domain.coordinates
    distance = np.linalg.norm(coordinates - np.asarray(center, dtype=np.float64), axis=1)
    values = np.where(distance <= radius, inclusion_speed, background_speed)
    description = (
        f"circular inclusion centered at ({center[0]:g}, {center[1]:g}), "
        f"radius={radius:g}, c_background={background_speed:g}, "
        f"c_inclusion={inclusion_speed:g}"
    )
    return _checked_medium(domain, values, description)


# ---------------------------------------------------------------------------
# Feature-composed phantom: a constant background plus additive smooth features
# (disk / ellipse / ring), each blended with a shared ``tanh`` sigmoid. This is
# the general vocabulary that makes richer tissue models — including the
# professor's breast phantom — expressible as data (and therefore as config).
# ---------------------------------------------------------------------------
def _sigmoid_inside(distance: NDArray[np.float64], sharpness: float) -> NDArray[np.float64]:
    """Smooth indicator: 1 well inside (distance < 0), 0 outside."""

    return 0.5 * (1.0 - np.tanh(sharpness * distance))


def _disk_delta(x, y, feature: Mapping[str, Any], sharpness: float) -> NDArray[np.float64]:
    cx, cy = feature["center"]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return float(feature["contrast"]) * _sigmoid_inside(r - float(feature["radius"]), sharpness)


def _ellipse_delta(x, y, feature: Mapping[str, Any], sharpness: float) -> NDArray[np.float64]:
    cx, cy = feature["center"]
    ax, ay = feature["semi_axes"]
    mask = ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2
    # mask == 1 is the ellipse boundary, so (mask - 1) is the signed argument.
    return float(feature["contrast"]) * _sigmoid_inside(mask - 1.0, sharpness)


def _ring_delta(x, y, feature: Mapping[str, Any], sharpness: float) -> NDArray[np.float64]:
    cx, cy = feature["center"]
    ring_radius = float(feature["radius"])
    half_width = float(feature["half_width"])
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # A rising edge at the inner wall times a falling edge at the outer wall,
    # each blended at half the base sharpness (matching the original phantom).
    edge = 0.5 * sharpness
    rising = 0.5 * (1.0 + np.tanh(edge * (r - (ring_radius - half_width))))
    falling = 0.5 * (1.0 - np.tanh(edge * (r - (ring_radius + half_width))))
    return float(feature["contrast"]) * rising * falling


_FEATURE_BUILDERS = {"disk": _disk_delta, "ellipse": _ellipse_delta, "ring": _ring_delta}


def feature_speed(
    domain: Domain,
    background_speed: float,
    sharpness: float,
    features: Sequence[Mapping[str, Any]],
) -> Medium:
    """Compose a nodal sound speed from additive smooth features on a background."""

    if not np.isfinite(background_speed) or background_speed <= 0.0:
        raise ValueError(f"background_speed must be positive and finite; got {background_speed!r}")
    if not np.isfinite(sharpness) or sharpness <= 0.0:
        raise ValueError(f"sharpness must be positive and finite; got {sharpness!r}")

    coordinates = domain.coordinates
    x, y = coordinates[:, 0], coordinates[:, 1]
    values = np.full(domain.basis.N, float(background_speed), dtype=np.float64)
    labels: list[str] = []
    for index, feature in enumerate(features):
        kind = str(feature.get("type", ""))
        builder = _FEATURE_BUILDERS.get(kind)
        if builder is None:
            supported = ", ".join(sorted(_FEATURE_BUILDERS))
            raise ValueError(
                f"feature {index} has unsupported type {kind!r}; supported: {supported}"
            )
        values = values + builder(x, y, feature, float(sharpness))
        labels.append(str(feature.get("label", kind)))

    description = (
        f"feature phantom (background c={background_speed:g}, sharpness={sharpness:g}, "
        f"features=[{', '.join(labels)}])"
    )
    return _checked_medium(domain, values, description)


# The professor's exact breast phantom (from the original alpha_expr prototype,
# verbatim), dimensionless with water background c = 1 and shared sharpness 80.
PROFESSOR_BREAST_PHANTOM: dict[str, Any] = {
    "background_speed": 1.0,
    "sharpness": 80.0,
    "features": (
        {"type": "ellipse", "label": "gland", "center": (0.0275, -0.0275),
         "semi_axes": (0.3025, 0.22), "contrast": 0.3},
        {"type": "disk", "label": "lesion_1_high_contrast", "center": (-0.165, 0.165),
         "radius": 0.066, "contrast": 0.8},
        {"type": "disk", "label": "lesion_2_cyst", "center": (0.1375, 0.0825),
         "radius": 0.044, "contrast": -0.35},
        {"type": "disk", "label": "lesion_3_resolution_test", "center": (0.055, -0.1925),
         "radius": 0.033, "contrast": 0.6},
        {"type": "ring", "label": "skin", "center": (0.0, 0.0),
         "radius": 0.55, "half_width": 0.04, "contrast": 0.25},
    ),
}

PHANTOM_PRESETS: dict[str, dict[str, Any]] = {"professor_breast": PROFESSOR_BREAST_PHANTOM}

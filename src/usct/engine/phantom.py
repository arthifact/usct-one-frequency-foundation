"""General feature-composed sound-speed phantom.

The professor's ``old_lowf_inv.py`` phantom (``alpha_expr``) is a constant
background with several smoothly superposed features: a glandular ellipse,
three lesions, and a skin ring, each blended with a ``tanh`` sigmoid of shared
``sharpness``. Rather than hard-code that one phantom, this module defines a
small feature vocabulary — ``disk``, ``ellipse``, ``ring`` — and composes an
arbitrary list of them additively. The professor's tissue then becomes one
preset (:data:`PROFESSOR_BREAST_PHANTOM`) that reproduces ``alpha_expr``
exactly, proving the medium contract is genuinely swappable and extensible.

All feature contrasts are additive on the background speed, matching the
original dimensionless construction (water background ``c = 1``).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usct.domain import Domain
from usct.medium import Medium, _checked_medium


def _sigmoid_inside(distance: NDArray[np.float64], sharpness: float) -> NDArray[np.float64]:
    """Smooth indicator that is 1 well inside (distance < 0) and 0 outside."""

    return 0.5 * (1.0 - np.tanh(sharpness * distance))


def _disk_delta(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    feature: Mapping[str, Any],
    sharpness: float,
) -> NDArray[np.float64]:
    cx, cy = feature["center"]
    radius = float(feature["radius"])
    contrast = float(feature["contrast"])
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return contrast * _sigmoid_inside(r - radius, sharpness)


def _ellipse_delta(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    feature: Mapping[str, Any],
    sharpness: float,
) -> NDArray[np.float64]:
    cx, cy = feature["center"]
    ax, ay = feature["semi_axes"]
    contrast = float(feature["contrast"])
    mask = ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2
    # The ellipse boundary is mask == 1, so (mask - 1) plays the role of a
    # signed distance for the shared sigmoid blend.
    return contrast * _sigmoid_inside(mask - 1.0, sharpness)


def _ring_delta(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    feature: Mapping[str, Any],
    sharpness: float,
) -> NDArray[np.float64]:
    cx, cy = feature["center"]
    ring_radius = float(feature["radius"])
    half_width = float(feature["half_width"])
    contrast = float(feature["contrast"])
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # A rising edge at the inner wall times a falling edge at the outer wall,
    # each blended at half the base sharpness exactly as in the original phantom.
    edge = 0.5 * sharpness
    rising = 0.5 * (1.0 + np.tanh(edge * (r - (ring_radius - half_width))))
    falling = 0.5 * (1.0 - np.tanh(edge * (r - (ring_radius + half_width))))
    return contrast * rising * falling


_FEATURE_BUILDERS = {
    "disk": _disk_delta,
    "ellipse": _ellipse_delta,
    "ring": _ring_delta,
}


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
    x = coordinates[:, 0]
    y = coordinates[:, 1]
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
        values = values + builder(x, y, feature, sharpness)
        labels.append(feature.get("label", kind))

    description = (
        f"feature phantom (background c={background_speed:g}, sharpness={sharpness:g}, "
        f"features=[{', '.join(labels)}])"
    )
    return _checked_medium(domain, values, description)


# --- The professor's exact breast phantom, ported verbatim from alpha_expr ----
# Dimensionless units: water background c = 1, contrasts additive, sharpness 80.
# Reproduces old_lowf_inv.py lines 193-230 feature for feature.
PROFESSOR_BREAST_PHANTOM: dict[str, Any] = {
    "background_speed": 1.0,
    "sharpness": 80.0,
    "features": (
        {
            "type": "ellipse",
            "label": "gland",
            "center": (0.0275, -0.0275),
            "semi_axes": (0.3025, 0.22),
            "contrast": 0.3,
        },
        {
            "type": "disk",
            "label": "lesion_1_high_contrast",
            "center": (-0.165, 0.165),
            "radius": 0.066,
            "contrast": 0.8,
        },
        {
            "type": "disk",
            "label": "lesion_2_cyst",
            "center": (0.1375, 0.0825),
            "radius": 0.044,
            "contrast": -0.35,
        },
        {
            "type": "disk",
            "label": "lesion_3_resolution_test",
            "center": (0.055, -0.1925),
            "radius": 0.033,
            "contrast": 0.6,
        },
        {
            "type": "ring",
            "label": "skin",
            "center": (0.0, 0.0),
            "radius": 0.55,
            "half_width": 0.04,
            "contrast": 0.25,
        },
    ),
}

PHANTOM_PRESETS: dict[str, dict[str, Any]] = {
    "professor_breast": PROFESSOR_BREAST_PHANTOM,
}

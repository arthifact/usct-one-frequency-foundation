"""Transparent deterministic sound-speed models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from usct.domain import Domain


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

"""Immutable simulation configuration and TOML loading."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from usct.physics.domain import build_disk
from usct.physics.forcing import parse_pattern_name
from usct.physics.medium import feature_speed


@dataclass(frozen=True)
class DomainConfig:
    """Circular-domain radius, uniform refinement count, and wavelength guard."""

    radius: float
    refinements: int
    minimum_points_per_wavelength: float = 12.0
    enforce_wavelength_resolution: bool = True


@dataclass(frozen=True)
class WaveConfig:
    """One temporal frequency in hertz and explicit dimensionless loss."""

    frequency: float
    loss: float = 0.0


@dataclass(frozen=True)
class MediumConfig:
    """Constant, circular-inclusion, or feature-phantom nodal sound-speed model."""

    kind: str
    background_speed: float
    inclusion_center: tuple[float, float] = (0.0, 0.0)
    inclusion_radius: float | None = None
    inclusion_speed: float | None = None
    sharpness: float | None = None
    features: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ForcingConfig:
    """Explicit ordered boundary-pressure pattern names."""

    patterns: tuple[str, ...]


@dataclass(frozen=True)
class SimulationConfig:
    """Complete immutable input for one Dirichlet-to-Neumann experiment."""

    domain: DomainConfig
    wave: WaveConfig
    medium: MediumConfig
    forcing: ForcingConfig
    units_system: str


@dataclass(frozen=True)
class WavelengthDiagnostics:
    """Actual mesh resolution relative to the shortest configured wavelength."""

    minimum_sound_speed: float
    wavelength_minimum: float
    h_max: float
    points_per_wavelength: float
    required_points_per_wavelength: float


def _table(data: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration requires a [{name}] table")
    return value


def _parse_feature(feature: object, index: int) -> dict[str, Any]:
    if not isinstance(feature, Mapping):
        raise ValueError(f"medium.feature[{index}] must be a table")
    kind = str(feature.get("type", ""))
    out: dict[str, Any] = {"type": kind, "label": str(feature.get("label", kind))}
    for key in ("center", "semi_axes"):
        if key in feature:
            pair = feature[key]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"medium.feature[{index}].{key} must contain two numbers")
            out[key] = (float(pair[0]), float(pair[1]))
    for key in ("radius", "half_width", "contrast"):
        if key in feature:
            out[key] = float(feature[key])
    return out


def _medium_from_table(table: Mapping[str, object]) -> MediumConfig:
    center_value = table.get("inclusion_center", (0.0, 0.0))
    if not isinstance(center_value, (list, tuple)) or len(center_value) != 2:
        raise ValueError("medium.inclusion_center must contain exactly two numbers")
    features_value = table.get("feature", [])
    if not isinstance(features_value, list):
        raise ValueError("medium.feature must be an array of [[medium.feature]] tables")
    return MediumConfig(
        kind=str(table.get("kind", "")),
        background_speed=float(table.get("background_speed", float("nan"))),
        inclusion_center=(float(center_value[0]), float(center_value[1])),
        inclusion_radius=(
            None if "inclusion_radius" not in table else float(table["inclusion_radius"])
        ),
        inclusion_speed=(
            None if "inclusion_speed" not in table else float(table["inclusion_speed"])
        ),
        sharpness=(None if "sharpness" not in table else float(table["sharpness"])),
        features=tuple(_parse_feature(item, i) for i, item in enumerate(features_value)),
    )


def _parse_config(data: Mapping[str, object]) -> SimulationConfig:
    domain = _table(data, "domain")
    wave = _table(data, "wave")
    forcing = _table(data, "forcing")
    units = _table(data, "units")
    refinements = domain.get("refinements")
    patterns = forcing.get("patterns")
    if isinstance(refinements, bool) or not isinstance(refinements, int):
        raise ValueError("domain.refinements must be an integer")
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        raise ValueError("forcing.patterns must be an array of strings")
    return SimulationConfig(
        domain=DomainConfig(
            radius=float(domain.get("radius", float("nan"))),
            refinements=refinements,
            minimum_points_per_wavelength=float(domain.get("minimum_points_per_wavelength", 12.0)),
        ),
        wave=WaveConfig(
            frequency=float(wave.get("frequency", float("nan"))),
            loss=float(wave.get("loss", 0.0)),
        ),
        medium=_medium_from_table(_table(data, "medium")),
        forcing=ForcingConfig(patterns=tuple(patterns)),
        units_system=str(units.get("system", "")),
    )


def _minimum_speed(config: SimulationConfig, domain) -> float:
    if config.medium.kind == "feature_phantom":
        medium = feature_speed(
            domain,
            config.medium.background_speed,
            config.medium.sharpness or 1.0,
            config.medium.features,
        )
        return float(np.min(medium.sound_speed))
    speeds = [config.medium.background_speed]
    if config.medium.inclusion_speed is not None:
        speeds.append(config.medium.inclusion_speed)
    return float(min(speeds))


def wavelength_diagnostics(config: SimulationConfig) -> WavelengthDiagnostics:
    """Calculate actual ``lambda_min / h_max`` for the generated disk mesh."""

    domain = build_disk(config.domain.radius, config.domain.refinements)
    minimum_speed = _minimum_speed(config, domain)
    wavelength = minimum_speed / config.wave.frequency
    return WavelengthDiagnostics(
        minimum_sound_speed=minimum_speed,
        wavelength_minimum=wavelength,
        h_max=domain.h_max,
        points_per_wavelength=wavelength / domain.h_max,
        required_points_per_wavelength=config.domain.minimum_points_per_wavelength,
    )


def validate_config(config: SimulationConfig) -> None:
    """Reject physically invalid or under-resolved one-frequency configurations."""

    positive = {
        "domain.radius": config.domain.radius,
        "wave.frequency": config.wave.frequency,
        "medium.background_speed": config.medium.background_speed,
        "domain.minimum_points_per_wavelength": config.domain.minimum_points_per_wavelength,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite; got {value!r}")
    if (
        isinstance(config.domain.refinements, bool)
        or not isinstance(config.domain.refinements, int)
        or config.domain.refinements < 0
    ):
        raise ValueError("domain.refinements must be nonnegative")
    if not np.isfinite(config.wave.loss) or config.wave.loss < 0.0:
        raise ValueError(f"wave.loss must be nonnegative and finite; got {config.wave.loss!r}")
    if config.medium.kind not in {"constant", "circular_inclusion", "feature_phantom"}:
        raise ValueError(f"unsupported medium kind {config.medium.kind!r}")
    _validate_inclusion(config.medium)
    _validate_feature_phantom(config.medium)
    if not config.forcing.patterns:
        raise ValueError("at least one forcing pattern is required")
    if len(set(config.forcing.patterns)) != len(config.forcing.patterns):
        raise ValueError("forcing pattern names must not be repeated")
    for pattern in config.forcing.patterns:
        parse_pattern_name(pattern)
    if config.units_system not in {"SI", "dimensionless"}:
        raise ValueError("units.system must be 'SI' or 'dimensionless'")
    if config.domain.enforce_wavelength_resolution:
        diagnostics = wavelength_diagnostics(config)
        if diagnostics.points_per_wavelength < diagnostics.required_points_per_wavelength:
            raise ValueError(
                "mesh resolution is insufficient: "
                f"lambda_min={diagnostics.wavelength_minimum:.6g}, "
                f"h_max={diagnostics.h_max:.6g}, "
                f"points_per_wavelength={diagnostics.points_per_wavelength:.3f}, "
                f"required={diagnostics.required_points_per_wavelength:.3f}"
            )


def _validate_inclusion(medium: MediumConfig) -> None:
    if medium.kind != "circular_inclusion":
        return
    if medium.inclusion_radius is None or medium.inclusion_speed is None:
        raise ValueError("circular_inclusion requires inclusion_radius and inclusion_speed")
    values = (*medium.inclusion_center, medium.inclusion_radius, medium.inclusion_speed)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("circular inclusion parameters must be finite")
    if medium.inclusion_radius <= 0.0 or medium.inclusion_speed <= 0.0:
        raise ValueError("inclusion radius and sound speed must be positive")


def _validate_feature_phantom(medium: MediumConfig) -> None:
    if medium.kind != "feature_phantom":
        return
    if medium.sharpness is None or not np.isfinite(medium.sharpness) or medium.sharpness <= 0.0:
        raise ValueError("feature_phantom requires a positive sharpness")
    if not medium.features:
        raise ValueError("feature_phantom requires at least one [[medium.feature]]")
    for index, feature in enumerate(medium.features):
        kind = feature.get("type")
        if kind not in {"disk", "ellipse", "ring"}:
            raise ValueError(f"medium.feature[{index}] has unsupported type {kind!r}")
        if "center" not in feature:
            raise ValueError(f"medium.feature[{index}] requires a center")
        if "contrast" not in feature:
            raise ValueError(f"medium.feature[{index}] requires a contrast")
        if kind in {"disk", "ring"} and "radius" not in feature:
            raise ValueError(f"medium.feature[{index}] ({kind}) requires a radius")
        if kind == "ring" and "half_width" not in feature:
            raise ValueError(f"medium.feature[{index}] (ring) requires a half_width")
        if kind == "ellipse" and "semi_axes" not in feature:
            raise ValueError(f"medium.feature[{index}] (ellipse) requires semi_axes")


def load_config(path: Path) -> SimulationConfig:
    """Load and validate a TOML simulation configuration from ``path``."""

    with Path(path).open("rb") as stream:
        data = tomllib.load(stream)
    config = _parse_config(data)
    validate_config(config)
    return config

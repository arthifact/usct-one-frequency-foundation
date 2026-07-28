"""Configuration parsing and validation tests."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from usct.io.config import load_config, validate_config


def test_valid_toml_loads_into_immutable_records() -> None:
    config = load_config(Path("configs/analytic_disk.toml"))
    assert config.domain.radius == 1.0
    assert config.forcing.patterns == ("exp:0", "exp:1", "exp:2")
    with pytest.raises(FrozenInstanceError):
        config.wave.frequency = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("domain", "radius", 0.0),
        ("wave", "frequency", -1.0),
        ("wave", "loss", -0.01),
        ("medium", "background_speed", 0.0),
    ),
)
def test_invalid_physical_values_are_rejected(section: str, field: str, value: float) -> None:
    config = load_config(Path("configs/analytic_disk.toml"))
    changed_section = replace(getattr(config, section), **{field: value})
    changed = replace(config, **{section: changed_section})
    with pytest.raises(ValueError):
        validate_config(changed)


@pytest.mark.parametrize("pattern", ("cos:-1", "sin:1.5", "exp:x", "point:1", " cos:1"))
def test_invalid_modes_are_rejected(pattern: str) -> None:
    config = load_config(Path("configs/analytic_disk.toml"))
    changed = replace(config, forcing=replace(config.forcing, patterns=(pattern,)))
    with pytest.raises(ValueError, match="unsupported boundary pattern"):
        validate_config(changed)


def test_repeated_patterns_and_unsupported_medium_are_rejected() -> None:
    config = load_config(Path("configs/analytic_disk.toml"))
    repeated = replace(
        config,
        forcing=replace(config.forcing, patterns=("exp:1", "exp:1")),
    )
    unsupported = replace(config, medium=replace(config.medium, kind="anatomy"))
    with pytest.raises(ValueError, match="must not be repeated"):
        validate_config(repeated)
    with pytest.raises(ValueError, match="unsupported medium"):
        validate_config(unsupported)


def test_wavelength_guard_reports_useful_numbers() -> None:
    config = load_config(Path("configs/analytic_disk.toml"))
    under_resolved = replace(
        config,
        domain=replace(config.domain, refinements=1),
        wave=replace(config.wave, frequency=20.0),
    )
    message = "lambda_min=.*h_max=.*points_per_wavelength=.*required="
    with pytest.raises(ValueError, match=message):
        validate_config(under_resolved)

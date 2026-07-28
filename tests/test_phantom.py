"""Feature-phantom medium: config parsing, validation, and construction."""

from pathlib import Path

import numpy as np
import pytest

from usct.dtn.simulation import simulate_dtn_one_frequency
from usct.io.config import load_config, validate_config
from usct.physics.domain import build_disk
from usct.physics.medium import PROFESSOR_BREAST_PHANTOM, feature_speed

PHANTOM = Path("configs/breast_phantom.toml")


def _alpha_expr(x, y, sharpness=80.0):
    r = np.hypot(x, y)
    c = np.ones_like(r)
    c += 0.3 * 0.5 * (1 - np.tanh(sharpness * (((x - 0.0275) / 0.3025) ** 2
                                               + ((y + 0.0275) / 0.22) ** 2 - 1)))
    c += 0.8 * 0.5 * (1 - np.tanh(sharpness * (np.hypot(x + 0.165, y - 0.165) - 0.066)))
    c += -0.35 * 0.5 * (1 - np.tanh(sharpness * (np.hypot(x - 0.1375, y - 0.0825) - 0.044)))
    c += 0.6 * 0.5 * (1 - np.tanh(sharpness * (np.hypot(x - 0.055, y + 0.1925) - 0.033)))
    rising = 0.5 * (1 + np.tanh(sharpness * 0.5 * (r - (0.55 - 0.04))))
    falling = 0.5 * (1 - np.tanh(sharpness * 0.5 * (r - (0.55 + 0.04))))
    return c + 0.25 * rising * falling


def test_phantom_config_parses_features() -> None:
    config = load_config(PHANTOM)
    assert config.medium.kind == "feature_phantom"
    assert config.medium.sharpness == 80.0
    assert len(config.medium.features) == 5
    assert {f["type"] for f in config.medium.features} == {"disk", "ellipse", "ring"}


def test_config_phantom_matches_alpha_expr() -> None:
    config = load_config(PHANTOM)
    domain = build_disk(config.domain.radius, 5)
    medium = feature_speed(
        domain, config.medium.background_speed, config.medium.sharpness, config.medium.features
    )
    x, y = domain.coordinates[:, 0], domain.coordinates[:, 1]
    assert np.max(np.abs(medium.sound_speed - _alpha_expr(x, y))) < 1e-12


def test_code_preset_matches_config() -> None:
    domain = build_disk(1.0, 4)
    preset = PROFESSOR_BREAST_PHANTOM
    from_preset = feature_speed(
        domain, preset["background_speed"], preset["sharpness"], preset["features"]
    )
    config = load_config(PHANTOM)
    from_config = feature_speed(
        domain, config.medium.background_speed, config.medium.sharpness, config.medium.features
    )
    assert np.array_equal(from_preset.sound_speed, from_config.sound_speed)


def test_phantom_config_simulates() -> None:
    config = load_config(PHANTOM)
    result = simulate_dtn_one_frequency(config)
    assert result.response.complex_response.shape[1] == len(config.forcing.patterns)
    assert np.all(np.isfinite(result.response.complex_response))


@pytest.mark.parametrize("mutation", [
    {"sharpness": 0.0},
    {"sharpness": None},
    {"features": ()},
])
def test_invalid_phantom_is_rejected(mutation: dict) -> None:
    from dataclasses import replace

    config = load_config(PHANTOM)
    changed = replace(config, medium=replace(config.medium, **mutation))
    with pytest.raises(ValueError):
        validate_config(changed)


def test_feature_with_unknown_type_is_rejected() -> None:
    from dataclasses import replace

    config = load_config(PHANTOM)
    bad = ({"type": "square", "center": (0.0, 0.0), "contrast": 0.1},)
    changed = replace(config, medium=replace(config.medium, features=bad))
    with pytest.raises(ValueError, match="unsupported type"):
        validate_config(changed)

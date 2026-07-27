"""Backwards-compatible re-export of the feature phantom.

The feature-composed phantom now lives in the foundation (:mod:`usct.medium`)
so it can be expressed as configuration and built by the verified pipeline.
This module remains as a stable import path for the engine layer.
"""

from usct.medium import (
    PHANTOM_PRESETS,
    PROFESSOR_BREAST_PHANTOM,
    feature_speed,
)

__all__ = ["PHANTOM_PRESETS", "PROFESSOR_BREAST_PHANTOM", "feature_speed"]

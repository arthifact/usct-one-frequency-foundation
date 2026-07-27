"""Medium-stage components: domain -> nodal sound speed.

Three interchangeable implementations of the same contract, in ascending
richness: a homogeneous background, a single circular inclusion, and the
general feature phantom that reproduces the professor's breast tissue. Because
all three satisfy one ``MediumModel`` signature, the pipeline and UI swap them
without knowing which is selected.
"""

from collections.abc import Mapping
from typing import Any

from usct.domain import Domain
from usct.engine.contracts import MEDIUM, Component, ParamSpec
from usct.engine.phantom import PHANTOM_PRESETS, feature_speed
from usct.engine.registry import register
from usct.medium import Medium, circular_inclusion, constant_speed


def _jsonify_features(features: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
    """Convert preset features (with tuple coordinates) to JSON-friendly lists."""

    out: list[dict[str, Any]] = []
    for feature in features:
        item = dict(feature)
        for key in ("center", "semi_axes"):
            if key in item:
                item[key] = list(item[key])
        out.append(item)
    return out


_PROFESSOR = PHANTOM_PRESETS["professor_breast"]


def _constant(domain: Domain, params: Mapping[str, Any]) -> Medium:
    return constant_speed(domain, float(params["background_speed"]))


def _circular_inclusion(domain: Domain, params: Mapping[str, Any]) -> Medium:
    center = params["center"]
    return circular_inclusion(
        domain,
        float(params["background_speed"]),
        (float(center[0]), float(center[1])),
        float(params["radius"]),
        float(params["inclusion_speed"]),
    )


def _feature_phantom(domain: Domain, params: Mapping[str, Any]) -> Medium:
    preset = str(params.get("preset", "professor_breast"))
    if preset != "custom" and preset in PHANTOM_PRESETS:
        spec = PHANTOM_PRESETS[preset]
        return feature_speed(
            domain, spec["background_speed"], spec["sharpness"], spec["features"]
        )
    return feature_speed(
        domain,
        float(params["background_speed"]),
        float(params["sharpness"]),
        params.get("features", ()),
    )


register(
    Component(
        stage=MEDIUM,
        name="constant",
        summary="Spatially constant sound speed (homogeneous background).",
        run=_constant,
        params=(
            ParamSpec(
                "background_speed",
                "Background speed",
                "float",
                1.0,
                minimum=1e-6,
                step=0.01,
                unit="speed",
                help="Uniform sound speed everywhere in the disk.",
            ),
        ),
    )
)

register(
    Component(
        stage=MEDIUM,
        name="circular_inclusion",
        summary="Two-speed disk with one circular demonstration inclusion.",
        run=_circular_inclusion,
        params=(
            ParamSpec(
                "background_speed", "Background speed", "float", 1.0, minimum=1e-6, unit="speed"
            ),
            ParamSpec("center", "Inclusion center", "vec2", [0.0, 0.0]),
            ParamSpec("radius", "Inclusion radius", "float", 0.3, minimum=1e-6, step=0.01),
            ParamSpec(
                "inclusion_speed", "Inclusion speed", "float", 1.5, minimum=1e-6, unit="speed"
            ),
        ),
    )
)

register(
    Component(
        stage=MEDIUM,
        name="feature_phantom",
        summary="Background plus additive smooth features (disk/ellipse/ring); "
        "reproduces the professor's breast tissue exactly.",
        run=_feature_phantom,
        params=(
            ParamSpec(
                "preset",
                "Preset",
                "choice",
                "professor_breast",
                choices=("professor_breast", "custom"),
                help="Select a built-in phantom, or 'custom' to use the fields below.",
            ),
            ParamSpec(
                "background_speed",
                "Background speed",
                "float",
                float(_PROFESSOR["background_speed"]),
                minimum=1e-6,
                unit="speed",
                help="Used only when preset is 'custom'.",
            ),
            ParamSpec(
                "sharpness",
                "Edge sharpness",
                "float",
                float(_PROFESSOR["sharpness"]),
                minimum=1.0,
                step=1.0,
                help="tanh transition steepness shared by every feature edge.",
            ),
            ParamSpec(
                "features",
                "Features",
                "features",
                _jsonify_features(_PROFESSOR["features"]),
                help="Ordered disk/ellipse/ring features; used only when preset is 'custom'.",
            ),
        ),
    )
)

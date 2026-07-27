"""Mesh-stage components: parameters -> discretized domain."""

from collections.abc import Mapping
from typing import Any

from usct.domain import Domain, build_disk
from usct.engine.contracts import MESH, Component, ParamSpec
from usct.engine.registry import register


def _uniform_disk(params: Mapping[str, Any]) -> Domain:
    return build_disk(float(params["radius"]), int(params["refinements"]))


register(
    Component(
        stage=MESH,
        name="uniform_disk",
        summary="Uniformly refined circular P1 triangle mesh, independent of the medium.",
        run=_uniform_disk,
        params=(
            ParamSpec(
                "radius",
                "Radius",
                "float",
                1.0,
                minimum=1e-3,
                maximum=10.0,
                step=0.01,
                unit="length",
                help="Disk radius in configured units.",
            ),
            ParamSpec(
                "refinements",
                "Refinements",
                "int",
                5,
                minimum=0,
                maximum=8,
                step=1,
                help=(
                    "Uniform refinement passes; each roughly quadruples the triangle count. "
                    "Refinement is deliberately truth-independent: it never tracks the medium, "
                    "so the mesh cannot leak the answer into a future inverse problem."
                ),
            ),
        ),
    )
)

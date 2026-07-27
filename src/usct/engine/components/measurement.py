"""Measurement-stage components: solved field -> observable boundary response."""

from collections.abc import Mapping
from typing import Any

from usct.domain import Domain
from usct.engine.contracts import MEASUREMENT, Component, ForwardField
from usct.engine.registry import register
from usct.measurement import BoundaryResponse, project_neumann_trace


def _weak_neumann(
    domain: Domain,
    forward: ForwardField,
    params: Mapping[str, Any],
) -> BoundaryResponse:
    return project_neumann_trace(domain, forward.operator, forward.solution)


register(
    Component(
        stage=MEASUREMENT,
        name="weak_neumann",
        summary="Weak boundary-mass projection of the outward normal flux D = I + iQ.",
        run=_weak_neumann,
        params=(),
    )
)

"""Forcing-stage components: domain -> boundary-pressure columns."""

from collections.abc import Mapping
from typing import Any

from usct.domain import Domain
from usct.engine.contracts import FORCING, Component, ParamSpec
from usct.engine.registry import register
from usct.forcing import BoundaryForcing, build_boundary_forcing


def _fourier_modes(domain: Domain, params: Mapping[str, Any]) -> BoundaryForcing:
    return build_boundary_forcing(domain, tuple(params["patterns"]))


register(
    Component(
        stage=FORCING,
        name="fourier_modes",
        summary="Named angular Fourier boundary patterns (constant, cos:m, sin:m, exp:m).",
        run=_fourier_modes,
        params=(
            ParamSpec(
                "patterns",
                "Boundary patterns",
                "patterns",
                ["constant", "cos:1", "sin:1", "cos:2", "sin:2"],
                help="Ordered, unique pattern names; one batched solve column per pattern.",
            ),
        ),
    )
)

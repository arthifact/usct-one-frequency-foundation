"""Forward-stage components: assemble and solve the frequency-domain problem."""

from collections.abc import Mapping
from math import pi
from typing import Any

from usct.domain import Domain
from usct.engine.contracts import FORWARD, Component, ForwardField, ParamSpec
from usct.engine.registry import register
from usct.forcing import BoundaryForcing
from usct.medium import Medium
from usct.operator import assemble_helmholtz
from usct.solver import solve_dirichlet


def _fem_p1_helmholtz(
    domain: Domain,
    medium: Medium,
    forcing: BoundaryForcing,
    params: Mapping[str, Any],
) -> ForwardField:
    frequency = float(params["frequency"])
    loss = float(params["loss"])
    omega = 2.0 * pi * frequency
    operator = assemble_helmholtz(domain, medium, omega, loss)
    solution = solve_dirichlet(domain, operator, forcing)
    return ForwardField(operator=operator, solution=solution)


register(
    Component(
        stage=FORWARD,
        name="fem_p1_helmholtz",
        summary="Complex P1 Helmholtz assembly with one batched LU over all patterns.",
        run=_fem_p1_helmholtz,
        params=(
            ParamSpec(
                "frequency",
                "Frequency",
                "float",
                0.477464829275686,
                minimum=1e-9,
                step=0.01,
                unit="Hz",
                help="Temporal frequency f; angular frequency is omega = 2 pi f. "
                "Default gives kR = 3 on the unit disk at c = 1.",
            ),
            ParamSpec(
                "loss",
                "Loss (eta)",
                "float",
                0.0,
                minimum=0.0,
                step=0.001,
                help="Dimensionless damping in k^2 = omega^2/c^2 (1 + i eta).",
            ),
        ),
    )
)

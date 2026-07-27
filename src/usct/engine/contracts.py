"""Stage contracts: the typed language every swappable component speaks.

This module defines *what* each pipeline stage promises to consume and produce,
independent of *how* any particular implementation does it. It is the port half
of a ports-and-adapters (hexagonal) design: the contracts live here; concrete
adapters register themselves in :mod:`usct.engine.registry`.

The five stages mirror the verified forward pipeline in
:func:`usct.simulation.simulate_dtn_one_frequency`::

    mesh -> medium -> forcing -> forward -> measurement

Each stage is a callable satisfying one ``Protocol`` below. Every component also
carries a JSON-serializable parameter schema (:class:`ParamSpec`) so a user
interface can render its controls without importing any physics code.

The design deliberately follows the pattern proven by SimPEG (interchangeable
``Simulation`` / ``Survey`` / ``Mapping`` base classes) and Devito/JUDI
(an abstracted forward operator): interfaces are stable, implementations are
free to change.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from usct.domain import Domain
from usct.forcing import BoundaryForcing
from usct.measurement import BoundaryResponse
from usct.medium import Medium
from usct.operator import HelmholtzOperator
from usct.solver import FieldSolution

# Stage identifiers. These are the contract names the registry is keyed on.
MESH = "mesh"
MEDIUM = "medium"
FORCING = "forcing"
FORWARD = "forward"
MEASUREMENT = "measurement"

STAGES: tuple[str, ...] = (MESH, MEDIUM, FORCING, FORWARD, MEASUREMENT)


@dataclass(frozen=True)
class ForwardField:
    """Interior solve output plus the operator that produced it.

    The measurement stage needs the assembled operator to recover the outward
    flux, so the forward stage hands both forward as a single artifact rather
    than forcing the pipeline to re-assemble.
    """

    operator: HelmholtzOperator
    solution: FieldSolution


@dataclass(frozen=True)
class ParamSpec:
    """One user-facing parameter of a component, renderable without physics.

    ``kind`` is a small closed vocabulary the UI knows how to draw:
    ``float``, ``int``, ``bool``, ``choice``, ``vec2``, ``patterns``
    (an ordered list of boundary-pattern names), or ``features`` (an ordered
    list of phantom features). Everything here is JSON-serializable via
    :meth:`to_dict` so the registry schema can be shipped to a browser.
    """

    name: str
    label: str
    kind: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    unit: str = ""
    help: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description for a user interface."""

        data: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
        }
        if self.minimum is not None:
            data["minimum"] = self.minimum
        if self.maximum is not None:
            data["maximum"] = self.maximum
        if self.step is not None:
            data["step"] = self.step
        if self.choices:
            data["choices"] = list(self.choices)
        if self.unit:
            data["unit"] = self.unit
        if self.help:
            data["help"] = self.help
        return data


@runtime_checkable
class MeshGenerator(Protocol):
    """Turn parameters into a discretized :class:`~usct.domain.Domain`."""

    def __call__(self, params: Mapping[str, Any]) -> Domain: ...


@runtime_checkable
class MediumModel(Protocol):
    """Assign a nodal sound speed on a given domain."""

    def __call__(self, domain: Domain, params: Mapping[str, Any]) -> Medium: ...


@runtime_checkable
class ForcingModel(Protocol):
    """Build the boundary-pressure columns driven on the domain boundary."""

    def __call__(self, domain: Domain, params: Mapping[str, Any]) -> BoundaryForcing: ...


@runtime_checkable
class ForwardModel(Protocol):
    """Assemble and solve the frequency-domain forward problem.

    ``params`` carries the excitation state that is neither mesh nor medium:
    ``frequency`` (hertz) and ``loss`` (dimensionless ``eta``).
    """

    def __call__(
        self,
        domain: Domain,
        medium: Medium,
        forcing: BoundaryForcing,
        params: Mapping[str, Any],
    ) -> ForwardField: ...


@runtime_checkable
class MeasurementModel(Protocol):
    """Extract the observable boundary response from a solved field."""

    def __call__(
        self,
        domain: Domain,
        forward: ForwardField,
        params: Mapping[str, Any],
    ) -> BoundaryResponse: ...


@dataclass(frozen=True)
class Component:
    """A named implementation of one stage contract, with its param schema.

    ``run`` is the callable satisfying the stage's ``Protocol``. The registry
    stores these; the pipeline resolves and calls them; the UI reads ``params``
    and ``summary`` to render a card.
    """

    stage: str
    name: str
    summary: str
    run: Any
    params: tuple[ParamSpec, ...] = field(default_factory=tuple)

    def defaults(self) -> dict[str, Any]:
        """Return a fresh ``{name: default}`` mapping for this component."""

        return {spec.name: spec.default for spec in self.params}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description for a user interface."""

        return {
            "stage": self.stage,
            "name": self.name,
            "summary": self.summary,
            "params": [spec.to_dict() for spec in self.params],
        }


def as_params(specs: Sequence[ParamSpec]) -> tuple[ParamSpec, ...]:
    """Normalize a parameter-spec sequence to an immutable tuple."""

    return tuple(specs)

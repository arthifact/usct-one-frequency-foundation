"""Staged pipeline that wires swappable components into one experiment.

A :class:`PipelineSpec` selects, per stage, which registered component runs and
with what parameters. :class:`PipelineRunner` executes the five stages in order
and — crucially for an interactive canvas — recomputes only the stages actually
affected by a change, using the true data dependencies:

    mesh      -> (medium, forcing, forward, measurement)
    medium    -> (forward, measurement)
    forcing   -> (forward, measurement)
    forward   -> (measurement)

So dragging the frequency slider re-solves forward and measurement but never
rebuilds the mesh; editing the medium leaves the forcing untouched.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from usct.domain import Domain
from usct.engine import registry
from usct.engine.contracts import (
    FORCING,
    FORWARD,
    MEASUREMENT,
    MEDIUM,
    MESH,
    STAGES,
    ForwardField,
)
from usct.forcing import BoundaryForcing
from usct.measurement import BoundaryResponse
from usct.medium import Medium

# Which upstream stages each stage's artifact truly depends on.
_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    MESH: (),
    MEDIUM: (MESH,),
    FORCING: (MESH,),
    FORWARD: (MESH, MEDIUM, FORCING),
    MEASUREMENT: (MESH, FORWARD),
}

# Default component per stage for a ready-to-run starting spec.
_DEFAULT_COMPONENT: dict[str, str] = {
    MESH: "uniform_disk",
    MEDIUM: "constant",
    FORCING: "fourier_modes",
    FORWARD: "fem_p1_helmholtz",
    MEASUREMENT: "weak_neumann",
}


@dataclass(frozen=True)
class StageChoice:
    """Selection of one component name plus a parameter override mapping."""

    component: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSpec:
    """A complete choice of component and parameters for every stage."""

    mesh: StageChoice
    medium: StageChoice
    forcing: StageChoice
    forward: StageChoice
    measurement: StageChoice

    def choice(self, stage: str) -> StageChoice:
        """Return the :class:`StageChoice` for ``stage``."""

        return getattr(self, stage)


@dataclass(frozen=True)
class StageReport:
    """What happened at one stage during a run."""

    stage: str
    component: str
    seconds: float
    recomputed: bool
    summary: str


@dataclass(frozen=True)
class PipelineResult:
    """Every stage artifact plus a per-stage execution report."""

    domain: Domain
    medium: Medium
    forcing: BoundaryForcing
    forward: ForwardField
    response: BoundaryResponse
    reports: tuple[StageReport, ...]


def default_spec(**overrides: StageChoice) -> PipelineSpec:
    """Build a runnable spec from registered defaults, with optional overrides."""

    choices = {stage: StageChoice(_DEFAULT_COMPONENT[stage]) for stage in STAGES}
    choices.update(overrides)
    return PipelineSpec(**choices)


def _resolved_params(stage: str, choice: StageChoice) -> dict[str, Any]:
    """Merge a component's declared defaults with the spec's overrides."""

    component = registry.get(stage, choice.component)
    params = component.defaults()
    params.update(dict(choice.params))
    return params


def _fingerprint(stage: str, choice: StageChoice) -> str:
    """A stable content hash of a stage selection, for change detection."""

    payload = {"component": choice.component, "params": _resolved_params(stage, choice)}
    return json.dumps(payload, sort_keys=True, default=str)


class PipelineRunner:
    """Stateful runner that caches artifacts and recomputes only what changed."""

    def __init__(self) -> None:
        self._fingerprints: dict[str, str] = {}
        self._artifacts: dict[str, Any] = {}

    def run(self, spec: PipelineSpec) -> PipelineResult:
        """Execute the pipeline, reusing cached artifacts where inputs are unchanged."""

        reports: list[StageReport] = []
        dirty: dict[str, bool] = {}
        for stage in STAGES:
            choice = spec.choice(stage)
            component = registry.get(stage, choice.component)
            params = _resolved_params(stage, choice)
            fingerprint = _fingerprint(stage, choice)

            upstream_dirty = any(dirty[dep] for dep in _DEPENDENCIES[stage])
            stale = fingerprint != self._fingerprints.get(stage)
            recompute = stale or upstream_dirty or stage not in self._artifacts
            dirty[stage] = recompute

            if recompute:
                started = perf_counter()
                self._artifacts[stage] = self._invoke(stage, component.run, params)
                seconds = perf_counter() - started
                self._fingerprints[stage] = fingerprint
            else:
                seconds = 0.0

            reports.append(
                StageReport(
                    stage=stage,
                    component=choice.component,
                    seconds=seconds,
                    recomputed=recompute,
                    summary=component.summary,
                )
            )

        return PipelineResult(
            domain=self._artifacts[MESH],
            medium=self._artifacts[MEDIUM],
            forcing=self._artifacts[FORCING],
            forward=self._artifacts[FORWARD],
            response=self._artifacts[MEASUREMENT],
            reports=tuple(reports),
        )

    def _invoke(self, stage: str, run: Any, params: dict[str, Any]) -> Any:
        if stage == MESH:
            return run(params)
        if stage == MEDIUM:
            return run(self._artifacts[MESH], params)
        if stage == FORCING:
            return run(self._artifacts[MESH], params)
        if stage == FORWARD:
            return run(
                self._artifacts[MESH],
                self._artifacts[MEDIUM],
                self._artifacts[FORCING],
                params,
            )
        return run(self._artifacts[MESH], self._artifacts[FORWARD], params)


def run_pipeline(spec: PipelineSpec) -> PipelineResult:
    """Run ``spec`` once with a fresh (uncached) runner."""

    return PipelineRunner().run(spec)

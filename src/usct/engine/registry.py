"""Name-based registry of swappable stage components.

Every implementation of a stage contract registers itself here under a short
name (``"constant"``, ``"feature_phantom"``, ...). Configuration files, the
pipeline, and any user interface then select an implementation purely by
``(stage, name)`` — never by importing a concrete function. This is the
indirection that lets a component be swapped, updated, or refined without any
caller knowing how it works.
"""

from collections.abc import Iterator

from usct.engine.contracts import STAGES, Component

_REGISTRY: dict[str, dict[str, Component]] = {stage: {} for stage in STAGES}


def register(component: Component) -> Component:
    """Register ``component`` under its stage and name; reject duplicates."""

    if component.stage not in _REGISTRY:
        raise ValueError(
            f"unknown stage {component.stage!r}; expected one of {', '.join(STAGES)}"
        )
    stage_table = _REGISTRY[component.stage]
    if component.name in stage_table:
        raise ValueError(
            f"component {component.name!r} is already registered for stage {component.stage!r}"
        )
    stage_table[component.name] = component
    return component


def get(stage: str, name: str) -> Component:
    """Return the registered component for ``(stage, name)`` or raise."""

    if stage not in _REGISTRY:
        raise ValueError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
    stage_table = _REGISTRY[stage]
    if name not in stage_table:
        available = ", ".join(sorted(stage_table)) or "none registered"
        raise ValueError(
            f"unknown {stage} component {name!r}; available: {available}"
        )
    return stage_table[name]


def available(stage: str) -> tuple[str, ...]:
    """Return the sorted names registered for ``stage``."""

    if stage not in _REGISTRY:
        raise ValueError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
    return tuple(sorted(_REGISTRY[stage]))


def components(stage: str) -> tuple[Component, ...]:
    """Return the components registered for ``stage`` in name order."""

    return tuple(_REGISTRY[stage][name] for name in available(stage))


def iter_all() -> Iterator[Component]:
    """Iterate every registered component across all stages."""

    for stage in STAGES:
        yield from components(stage)


def schema() -> dict[str, list[dict]]:
    """Return the full registry as JSON-serializable data for a user interface.

    The returned structure is ``{stage: [component_dict, ...]}`` and is the
    single source of truth a canvas or web client reads to render every stage,
    every selectable implementation, and every parameter control.
    """

    return {stage: [c.to_dict() for c in components(stage)] for stage in STAGES}

"""Swappable-component engine over the verified one-frequency foundation.

Importing this package registers every built-in component, so ``registry`` is
fully populated as soon as ``usct.engine`` is imported. The verified numerics in
``usct`` are untouched; every component here is a thin adapter that delegates to
them, which is exactly what keeps the foundation trustworthy while the engine
layer stays free to grow.
"""

# Importing the components package runs every register(...) call as a side
# effect, so the registry is fully populated on first import of this package.
from usct.engine import components as _components  # noqa: F401  (registration side effects)
from usct.engine import registry
from usct.engine.contracts import (
    STAGES,
    Component,
    ForwardField,
    ParamSpec,
)
from usct.engine.pipeline import (
    PipelineResult,
    PipelineRunner,
    PipelineSpec,
    StageChoice,
    StageReport,
    default_spec,
    run_pipeline,
)

__all__ = [
    "STAGES",
    "Component",
    "ForwardField",
    "ParamSpec",
    "PipelineResult",
    "PipelineRunner",
    "PipelineSpec",
    "StageChoice",
    "StageReport",
    "default_spec",
    "registry",
    "run_pipeline",
]

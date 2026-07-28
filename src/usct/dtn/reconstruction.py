"""Frequency-continuation complex-L2 FWI driver over the verified adjoint.

This is the "conventional control" reconstruction called for in ``AGENTS.md``
step 6: bounded L-BFGS-B on the boundary-flux misfit, using the adjoint-state
gradient from :mod:`usct.dtn.inversion`, with low-to-high frequency continuation
(each stage warm-starts from the previous stage's recovered field).

It is a research driver, not part of the verified forward foundation, and does
not modify any verified numerics.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from skfem import asm

from usct.dtn.inversion import objective_and_gradient
from usct.physics.boundary import boundary_mass_matrix
from usct.physics.domain import Domain
from usct.physics.forcing import build_boundary_forcing
from usct.physics.operator import _stiffness_form


@dataclass(frozen=True)
class Stage:
    """One frequency-continuation stage: a frequency, its modes, and controls."""

    frequency: float
    patterns: tuple[str, ...]
    loss: float = 0.0
    alpha: float = 0.0
    max_iterations: int = 30


@dataclass(frozen=True)
class StageReport:
    frequency: float
    pattern_count: int
    initial_misfit: float
    final_misfit: float
    final_data_misfit: float
    iterations: int


@dataclass(frozen=True)
class ReconstructionResult:
    sound_speed: NDArray[np.float64]
    reports: tuple[StageReport, ...]
    misfit_history: tuple[float, ...]


def reconstruct(
    domain: Domain,
    observed: Mapping[int, NDArray[np.complex128]],
    stages: Sequence[Stage],
    *,
    initial_speed: NDArray[np.float64],
    speed_bounds: tuple[float, float],
    background_speed: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> ReconstructionResult:
    """Run frequency-continuation FWI and return the recovered nodal sound speed.

    ``observed[i]`` is the target flux for ``stages[i]`` with shape
    ``(boundary_dofs, len(stages[i].patterns))`` in ascending boundary-DOF order.
    """

    omega_factor = 2.0 * np.pi
    boundary = domain.boundary_dofs
    # M_b and K do not depend on c: assemble once, reuse across every eval.
    boundary_mass = boundary_mass_matrix(domain)
    stiffness = asm(_stiffness_form, domain.basis).tocsc()

    speed = np.asarray(initial_speed, dtype=np.float64).copy()
    low, high = speed_bounds
    bounds = [(low, high)] * domain.basis.N
    if background_speed is not None:
        speed[boundary] = background_speed
        for dof in boundary:
            bounds[int(dof)] = (background_speed, background_speed)

    reports: list[StageReport] = []
    misfit_history: list[float] = []

    for index, stage in enumerate(stages):
        forcing = build_boundary_forcing(domain, stage.patterns)
        observed_flux = np.asarray(observed[index], dtype=np.complex128)
        omega = omega_factor * stage.frequency

        def evaluate(candidate: NDArray[np.float64], _stage=stage, _forcing=forcing,
                     _observed=observed_flux, _omega=omega):
            result = objective_and_gradient(
                domain, candidate, _forcing, _omega, _stage.loss, _observed,
                alpha=_stage.alpha, background_speed=background_speed,
                boundary_mass=boundary_mass, stiffness=stiffness,
            )
            misfit_history.append(result.misfit)
            return result.misfit, result.gradient

        initial_misfit, _ = evaluate(speed)
        optimized = minimize(
            evaluate, speed, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": stage.max_iterations, "ftol": 1e-12, "gtol": 1e-9},
        )
        speed = np.clip(optimized.x, low, high)
        final = objective_and_gradient(
            domain, speed, forcing, omega, stage.loss, observed_flux,
            alpha=0.0, background_speed=background_speed,
            boundary_mass=boundary_mass, stiffness=stiffness,
        )
        reports.append(
            StageReport(
                frequency=stage.frequency,
                pattern_count=len(stage.patterns),
                initial_misfit=initial_misfit,
                final_misfit=optimized.fun,
                final_data_misfit=final.data_misfit,
                iterations=int(optimized.nit),
            )
        )
        if progress is not None:
            progress(
                f"stage {index + 1}/{len(stages)}  f={stage.frequency:.3f}  "
                f"kR~{omega:.1f}  modes={len(stage.patterns)}  "
                f"data-misfit {initial_misfit:.3e} -> {final.data_misfit:.3e}  "
                f"({optimized.nit} iters)"
            )

    return ReconstructionResult(
        sound_speed=speed,
        reports=tuple(reports),
        misfit_history=tuple(misfit_history),
    )


def adaptive_patterns(
    frequency: float, radius: float = 1.0, minimum_mode: int = 2
) -> tuple[str, ...]:
    """Boundary patterns up to ``m_max = ceil(kR)`` at ``c = 1`` (constant + cos/sin)."""

    k_radius = 2.0 * np.pi * frequency * radius
    max_mode = max(minimum_mode, int(np.ceil(k_radius)))
    patterns = ["constant"]
    for mode in range(1, max_mode + 1):
        patterns.append(f"cos:{mode}")
        patterns.append(f"sin:{mode}")
    return tuple(patterns)

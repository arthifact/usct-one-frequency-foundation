"""Homogeneous-disk Bessel trust tests."""

import numpy as np
from scipy.special import jv

from usct.dtn.simulation import SimulationResult
from usct.dtn.verification import ANALYTIC_KR, AnalyticMetrics


def test_bessel_denominators_are_nonresonant(
    analytic_case: tuple[SimulationResult, AnalyticMetrics],
) -> None:
    _, metrics = analytic_case
    denominators = np.asarray([jv(mode, ANALYTIC_KR) for mode in metrics.modes])
    assert np.all(np.abs(denominators) > 0.1)


def test_analytic_field_and_flux_meet_declared_tolerances(
    analytic_case: tuple[SimulationResult, AnalyticMetrics],
) -> None:
    _, metrics = analytic_case
    assert max(metrics.field_errors) < 0.02
    assert max(metrics.flux_errors) < 0.10


def test_neumann_trace_has_outward_sign(
    analytic_case: tuple[SimulationResult, AnalyticMetrics],
) -> None:
    _, metrics = analytic_case
    assert min(metrics.flux_orientation) > 0.90

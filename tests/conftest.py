"""Shared numerical fixtures for the USCT foundation tests."""

from dataclasses import replace
from pathlib import Path

import pytest

from usct.config import SimulationConfig, load_config
from usct.domain import Domain, build_disk
from usct.medium import Medium, constant_speed
from usct.operator import HelmholtzOperator, assemble_helmholtz
from usct.simulation import SimulationResult, simulate_dtn_one_frequency
from usct.verification import AnalyticMetrics, run_analytic_disk


@pytest.fixture(scope="session")
def small_domain() -> Domain:
    return build_disk(radius=1.0, refinements=3)


@pytest.fixture(scope="session")
def small_medium(small_domain: Domain) -> Medium:
    return constant_speed(small_domain, speed=1.5)


@pytest.fixture(scope="session")
def small_operator(small_domain: Domain, small_medium: Medium) -> HelmholtzOperator:
    return assemble_helmholtz(small_domain, small_medium, omega=2.0, loss=0.0)


@pytest.fixture(scope="session")
def analytic_case() -> tuple[SimulationResult, AnalyticMetrics]:
    return run_analytic_disk(refinements=5)


@pytest.fixture(scope="session")
def compact_result() -> SimulationResult:
    config = load_config(Path("configs/analytic_disk.toml"))
    compact_domain = replace(
        config.domain,
        refinements=3,
        enforce_wavelength_resolution=False,
    )
    compact_forcing = replace(config.forcing, patterns=("exp:0", "exp:1"))
    compact_config = SimulationConfig(
        domain=compact_domain,
        wave=config.wave,
        medium=config.medium,
        forcing=compact_forcing,
        units_system=config.units_system,
    )
    return simulate_dtn_one_frequency(compact_config)

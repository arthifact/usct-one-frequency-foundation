"""Top-level orchestration of the boundary-driven one-frequency experiment."""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from math import pi
from platform import python_version
from typing import Any

from usct.config import SimulationConfig, validate_config
from usct.domain import Domain, build_disk
from usct.forcing import BoundaryForcing, build_boundary_forcing
from usct.measurement import BoundaryResponse, project_neumann_trace
from usct.medium import Medium, circular_inclusion, constant_speed
from usct.operator import assemble_helmholtz
from usct.solver import FieldSolution, solve_dirichlet

TIME_CONVENTION = "exp(-i omega t)"
MEASUREMENT_TYPE = "dirichlet_to_neumann"

MODELING_ASSUMPTIONS = (
    "two spatial dimensions",
    "scalar acoustic pressure",
    "constant density",
    "isotropic sound speed",
    "circular domain",
    "full boundary access",
    "known boundary pressure",
    "ideal coherent I/Q demodulation",
    "k_squared = omega^2 / c^2 * (1 + i eta)",
    "no transducer impulse response or calibration error",
    "no out-of-plane propagation",
)


@dataclass(frozen=True)
class SimulationResult:
    """Complete in-memory record of one idealized DtN forward experiment."""

    config: SimulationConfig
    domain: Domain
    medium: Medium
    forcing: BoundaryForcing
    solution: FieldSolution
    response: BoundaryResponse
    metadata: Mapping[str, Any]


def _build_medium(config: SimulationConfig, domain: Domain) -> Medium:
    medium = config.medium
    if medium.kind == "constant":
        return constant_speed(domain, medium.background_speed)
    return circular_inclusion(
        domain,
        medium.background_speed,
        medium.inclusion_center,
        medium.inclusion_radius or 0.0,
        medium.inclusion_speed or 0.0,
    )


def _dependency_versions() -> dict[str, str]:
    packages = ("numpy", "scipy", "scikit-fem", "matplotlib")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _metadata(
    config: SimulationConfig,
    domain: Domain,
    omega: float,
    medium: Medium,
) -> dict[str, Any]:
    wavelength = float(min(medium.sound_speed) / config.wave.frequency)
    return {
        "schema_version": 1,
        "measurement_type": MEASUREMENT_TYPE,
        "time_convention": TIME_CONVENTION,
        "frequency": config.wave.frequency,
        "angular_frequency": omega,
        "radius": config.domain.radius,
        "loss": config.wave.loss,
        "units": config.units_system,
        "h_max": domain.h_max,
        "minimum_wavelength": wavelength,
        "points_per_wavelength": wavelength / domain.h_max,
        "required_points_per_wavelength": config.domain.minimum_points_per_wavelength,
        "degrees_of_freedom": int(domain.basis.N),
        "triangles": int(domain.triangles.shape[0]),
        "medium_description": medium.description,
        "modeling_assumptions": list(MODELING_ASSUMPTIONS),
        "dependency_versions": _dependency_versions(),
        "python_version": python_version(),
    }


def simulate_dtn_one_frequency(config: SimulationConfig) -> SimulationResult:
    """Drive boundary pressure and return the complex outward boundary flux."""

    validate_config(config)
    domain = build_disk(config.domain.radius, config.domain.refinements)
    medium = _build_medium(config, domain)
    omega = 2.0 * pi * config.wave.frequency
    operator = assemble_helmholtz(domain, medium, omega, config.wave.loss)
    forcing = build_boundary_forcing(domain, config.forcing.patterns)
    solution = solve_dirichlet(domain, operator, forcing)
    response = project_neumann_trace(domain, operator, solution)
    return SimulationResult(
        config=config,
        domain=domain,
        medium=medium,
        forcing=forcing,
        solution=solution,
        response=response,
        metadata=_metadata(config, domain, omega, medium),
    )

"""Independent numerical trust checks for the forward foundation."""

from dataclasses import dataclass
from math import pi

import numpy as np
from numpy.typing import NDArray
from scipy.special import jv, jvp
from skfem import BilinearForm, asm

from usct.dtn.measurement import project_neumann_trace
from usct.dtn.simulation import SimulationResult, simulate_dtn_one_frequency
from usct.dtn.solver import solve_dirichlet
from usct.io.config import (
    DomainConfig,
    ForcingConfig,
    MediumConfig,
    SimulationConfig,
    WaveConfig,
)
from usct.physics.boundary import boundary_mass_matrix
from usct.physics.forcing import BoundaryForcing, build_boundary_forcing
from usct.physics.operator import assemble_helmholtz

ANALYTIC_KR = 3.0
ANALYTIC_MODES = (0, 1, 2)


@BilinearForm
def _domain_mass_form(u, v, _):
    return u * v


@dataclass(frozen=True)
class AnalyticMetrics:
    """Mass-weighted homogeneous-disk errors for several Fourier modes."""

    refinements: int
    degrees_of_freedom: int
    h_max: float
    points_per_wavelength: float
    modes: tuple[int, ...]
    field_errors: NDArray[np.float64]
    flux_errors: NDArray[np.float64]
    flux_orientation: NDArray[np.float64]


@dataclass(frozen=True)
class VerificationCheck:
    """One thresholded verification metric."""

    name: str
    value: float
    threshold: float
    passed: bool
    relation: str = "<"


@dataclass(frozen=True)
class VerificationSummary:
    """All CLI verification checks and detailed analytic convergence data."""

    checks: tuple[VerificationCheck, ...]
    analytic: AnalyticMetrics
    convergence: tuple[AnalyticMetrics, ...]

    @property
    def passed(self) -> bool:
        """Whether every required numerical check passed."""

        return all(check.passed for check in self.checks)


def _analytic_config(
    refinements: int,
    modes: tuple[int, ...],
    enforce_guard: bool,
) -> SimulationConfig:
    return SimulationConfig(
        domain=DomainConfig(
            radius=1.0,
            refinements=refinements,
            minimum_points_per_wavelength=12.0,
            enforce_wavelength_resolution=enforce_guard,
        ),
        wave=WaveConfig(frequency=ANALYTIC_KR / (2.0 * pi), loss=0.0),
        medium=MediumConfig(kind="constant", background_speed=1.0),
        forcing=ForcingConfig(tuple(f"exp:{mode}" for mode in modes)),
        units_system="dimensionless",
    )


def _mass_relative_error(
    numerical: NDArray[np.complex128],
    reference: NDArray[np.complex128],
    mass,
) -> float:
    error = numerical - reference
    numerator = float(np.real(np.vdot(error, mass @ error)))
    denominator = float(np.real(np.vdot(reference, mass @ reference)))
    return float(np.sqrt(max(numerator, 0.0) / denominator))


def _mass_orientation(
    numerical: NDArray[np.complex128],
    reference: NDArray[np.complex128],
    mass,
) -> float:
    cross = float(np.real(np.vdot(reference, mass @ numerical)))
    norm_numerical = float(np.real(np.vdot(numerical, mass @ numerical)))
    norm_reference = float(np.real(np.vdot(reference, mass @ reference)))
    return cross / np.sqrt(norm_numerical * norm_reference)


def _analytic_metrics(result: SimulationResult, modes: tuple[int, ...]) -> AnalyticMetrics:
    domain = result.domain
    radius = result.config.domain.radius
    speed = result.config.medium.background_speed
    wave_number = float(result.metadata["angular_frequency"]) / speed
    coordinates = domain.coordinates
    radii = np.linalg.norm(coordinates, axis=1)
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    domain_mass = asm(_domain_mass_form, domain.basis).tocsc()
    boundary_mass = boundary_mass_matrix(domain)
    positions = np.searchsorted(domain.boundary_dofs, result.response.boundary_dofs)
    sorted_boundary_mass = boundary_mass[positions][:, positions]
    field_errors: list[float] = []
    flux_errors: list[float] = []
    orientation: list[float] = []
    for column, mode in enumerate(modes):
        denominator = jv(mode, wave_number * radius)
        if abs(denominator) <= 0.1:
            raise ValueError(
                f"analytic Bessel denominator J_{mode}(kR)={denominator:.6g} is too near zero"
            )
        exact_field = jv(mode, wave_number * radii) / denominator * np.exp(1j * mode * angles)
        coefficient = wave_number * jvp(mode, wave_number * radius) / denominator
        exact_flux = coefficient * np.exp(1j * mode * result.response.theta)
        field_errors.append(
            _mass_relative_error(result.solution.field[:, column], exact_field, domain_mass)
        )
        numerical_flux = result.response.complex_response[:, column]
        flux_errors.append(_mass_relative_error(numerical_flux, exact_flux, sorted_boundary_mass))
        orientation.append(_mass_orientation(numerical_flux, exact_flux, sorted_boundary_mass))
    return AnalyticMetrics(
        refinements=result.config.domain.refinements,
        degrees_of_freedom=domain.basis.N,
        h_max=domain.h_max,
        points_per_wavelength=float(result.metadata["points_per_wavelength"]),
        modes=modes,
        field_errors=np.asarray(field_errors, dtype=np.float64),
        flux_errors=np.asarray(flux_errors, dtype=np.float64),
        flux_orientation=np.asarray(orientation, dtype=np.float64),
    )


def run_analytic_disk(
    refinements: int = 5,
    modes: tuple[int, ...] = ANALYTIC_MODES,
    *,
    enforce_wavelength_guard: bool = True,
) -> tuple[SimulationResult, AnalyticMetrics]:
    """Solve and compare the homogeneous disk against exact Bessel solutions."""

    result = simulate_dtn_one_frequency(
        _analytic_config(refinements, modes, enforce_wavelength_guard)
    )
    return result, _analytic_metrics(result, modes)


def _relative_array_error(numerical, reference) -> float:
    denominator = max(np.linalg.norm(reference), np.finfo(np.float64).eps)
    return float(np.linalg.norm(numerical - reference) / denominator)


def _linearity_metrics(result: SimulationResult) -> tuple[float, float, float, float]:
    domain = result.domain
    operator = assemble_helmholtz(
        domain,
        result.medium,
        float(result.metadata["angular_frequency"]),
        result.config.wave.loss,
    )
    basis_forcing = build_boundary_forcing(domain, ("cos:1", "sin:1", "exp:1"))
    basis_solution = solve_dirichlet(domain, operator, basis_forcing)
    basis_response = project_neumann_trace(domain, operator, basis_solution)
    scalar_a = 0.7 + 0.2j
    scalar_b = -0.3 + 0.8j
    combined_values = (
        scalar_a * basis_forcing.values[:, [0]] + scalar_b * basis_forcing.values[:, [1]]
    )
    combined = BoundaryForcing(("linear_combination",), combined_values)
    combined_solution = solve_dirichlet(domain, operator, combined)
    combined_response = project_neumann_trace(domain, operator, combined_solution)
    expected_field = (
        scalar_a * basis_solution.field[:, [0]] + scalar_b * basis_solution.field[:, [1]]
    )
    expected_flux = (
        scalar_a * basis_response.complex_response[:, [0]]
        + scalar_b * basis_response.complex_response[:, [1]]
    )
    field_linearity = _relative_array_error(combined_solution.field, expected_field)
    flux_linearity = _relative_array_error(
        combined_response.complex_response,
        expected_flux,
    )
    exp_field = _relative_array_error(
        basis_solution.field[:, 2],
        basis_solution.field[:, 0] + 1j * basis_solution.field[:, 1],
    )
    exp_flux = _relative_array_error(
        basis_response.complex_response[:, 2],
        basis_response.complex_response[:, 0] + 1j * basis_response.complex_response[:, 1],
    )
    return field_linearity, flux_linearity, exp_field, exp_flux


def _less(name: str, value: float, threshold: float) -> VerificationCheck:
    return VerificationCheck(name, value, threshold, value < threshold)


def run_verification() -> VerificationSummary:
    """Run analytic, algebraic, linearity, and three-mesh convergence checks."""

    result, analytic = run_analytic_disk()
    field_linearity, flux_linearity, exp_field, exp_flux = _linearity_metrics(result)
    convergence = tuple(
        run_analytic_disk(
            refinement,
            enforce_wavelength_guard=False,
        )[1]
        for refinement in (3, 4, 5)
    )
    field_sequence = np.asarray([max(item.field_errors) for item in convergence])
    flux_sequence = np.asarray([max(item.flux_errors) for item in convergence])
    field_ratio = float(field_sequence[-1] / field_sequence[0])
    flux_ratio = float(flux_sequence[-1] / flux_sequence[0])
    field_monotone = bool(np.all(np.diff(field_sequence) < 0.0))
    flux_monotone = bool(np.all(np.diff(flux_sequence) < 0.0))
    checks = (
        _less("analytic field error", float(max(analytic.field_errors)), 0.02),
        _less("analytic flux error", float(max(analytic.flux_errors)), 0.10),
        VerificationCheck(
            "outward Neumann sign",
            float(min(analytic.flux_orientation)),
            0.90,
            bool(min(analytic.flux_orientation) > 0.90),
            ">",
        ),
        _less(
            "interior residual",
            float(max(result.solution.report.normalized_interior_residuals)),
            1e-10,
        ),
        _less(
            "boundary enforcement",
            result.solution.report.maximum_boundary_value_error,
            1e-12,
        ),
        _less("field linearity", field_linearity, 1e-10),
        _less("flux linearity", flux_linearity, 1e-10),
        _less("exp field identity", exp_field, 1e-10),
        _less("exp flux identity", exp_flux, 1e-10),
        VerificationCheck(
            "field convergence ratio",
            field_ratio,
            0.60,
            field_monotone and field_ratio < 0.60,
        ),
        VerificationCheck(
            "flux convergence ratio",
            flux_ratio,
            0.60,
            flux_monotone and flux_ratio < 0.60,
        ),
    )
    return VerificationSummary(checks=checks, analytic=analytic, convergence=convergence)

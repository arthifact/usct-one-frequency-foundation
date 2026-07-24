"""Safe NPZ/JSON result serialization without pickle."""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from usct.simulation import SimulationResult


@dataclass(frozen=True)
class SavedResult:
    """Numeric saved arrays plus readable JSON metadata."""

    coordinates: NDArray[np.float64]
    triangles: NDArray[np.int64]
    boundary_dofs: NDArray[np.int64]
    boundary_theta: NDArray[np.float64]
    sound_speed: NDArray[np.float64]
    forcing_values: NDArray[np.complex128]
    fields: NDArray[np.complex128]
    boundary_flux_response: NDArray[np.complex128]
    normalized_interior_residuals: NDArray[np.float64]
    metadata: Mapping[str, Any]

    @property
    def forcing_names(self) -> tuple[str, ...]:
        """Forcing names in numeric-array column order."""

        return tuple(self.metadata["forcing_names"])


def _serialization_metadata(result: SimulationResult) -> dict[str, Any]:
    report = result.solution.report
    metadata = dict(result.metadata)
    metadata["forcing_names"] = list(result.forcing.names)
    metadata["solve_report"] = {
        key: value
        for key, value in asdict(report).items()
        if key != "normalized_interior_residuals"
    }
    metadata["array_shapes"] = {
        "coordinates": list(result.domain.coordinates.shape),
        "triangles": list(result.domain.triangles.shape),
        "forcing_values": list(result.forcing.values.shape),
        "fields": list(result.solution.field.shape),
        "boundary_flux_response": list(result.response.complex_response.shape),
    }
    return metadata


def save_result(result: SimulationResult, output_dir: Path) -> None:
    """Write complex numerical arrays to NPZ and readable metadata to JSON."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    positions = np.searchsorted(result.domain.boundary_dofs, result.response.boundary_dofs)
    if not np.array_equal(
        result.domain.boundary_dofs[positions],
        result.response.boundary_dofs,
    ):
        raise ValueError("response boundary indices do not match the simulation domain")
    forcing_sorted = result.forcing.values[positions]
    np.savez_compressed(
        output / "result.npz",
        coordinates=result.domain.coordinates,
        triangles=result.domain.triangles,
        boundary_dofs=result.response.boundary_dofs,
        boundary_theta=result.response.theta,
        sound_speed=result.medium.sound_speed,
        forcing_values=forcing_sorted,
        fields=result.solution.field,
        boundary_flux_response=result.response.complex_response,
        normalized_interior_residuals=result.solution.report.normalized_interior_residuals,
    )
    metadata = _serialization_metadata(result)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_result(output_dir: Path) -> SavedResult:
    """Load a saved simulation with NumPy pickle support explicitly disabled."""

    output = Path(output_dir)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    required_metadata = {
        "measurement_type",
        "time_convention",
        "frequency",
        "angular_frequency",
        "radius",
        "loss",
        "units",
        "dependency_versions",
        "forcing_names",
        "solve_report",
    }
    missing = required_metadata - metadata.keys()
    if missing:
        raise ValueError(f"metadata is missing required keys: {sorted(missing)}")
    with np.load(output / "result.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required_arrays = {
        "coordinates",
        "triangles",
        "boundary_dofs",
        "boundary_theta",
        "sound_speed",
        "forcing_values",
        "fields",
        "boundary_flux_response",
        "normalized_interior_residuals",
    }
    missing_arrays = required_arrays - arrays.keys()
    if missing_arrays:
        raise ValueError(f"result archive is missing arrays: {sorted(missing_arrays)}")
    return SavedResult(metadata=metadata, **arrays)

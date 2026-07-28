"""Result persistence and saved-result plotting tests."""

import json
from pathlib import Path

import numpy as np

from usct.dtn.simulation import SimulationResult
from usct.io.plotting import plot_result
from usct.io.results import load_result, save_result


def test_save_load_round_trip_preserves_complex_data(
    tmp_path: Path,
    compact_result: SimulationResult,
) -> None:
    save_result(compact_result, tmp_path)
    saved = load_result(tmp_path)
    positions = np.searchsorted(
        compact_result.domain.boundary_dofs,
        compact_result.response.boundary_dofs,
    )
    np.testing.assert_array_equal(
        saved.forcing_values,
        compact_result.forcing.values[positions],
    )
    np.testing.assert_array_equal(saved.fields, compact_result.solution.field)
    np.testing.assert_array_equal(
        saved.boundary_flux_response,
        compact_result.response.complex_response,
    )
    assert saved.fields.dtype == np.complex128
    assert saved.boundary_flux_response.dtype == np.complex128


def test_archive_needs_no_pickle_and_metadata_is_complete(
    tmp_path: Path,
    compact_result: SimulationResult,
) -> None:
    save_result(compact_result, tmp_path)
    with np.load(tmp_path / "result.npz", allow_pickle=False) as archive:
        assert archive.files
        assert all(archive[name].dtype.kind not in {"O", "S", "U"} for name in archive.files)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    required = {
        "measurement_type",
        "time_convention",
        "modeling_assumptions",
        "dependency_versions",
        "solve_report",
        "frequency",
        "angular_frequency",
        "radius",
        "loss",
        "units",
    }
    assert required <= metadata.keys()
    assert metadata["measurement_type"] == "dirichlet_to_neumann"
    assert metadata["time_convention"] == "exp(-i omega t)"


def test_plotting_saved_result_produces_nonempty_png(
    tmp_path: Path,
    compact_result: SimulationResult,
) -> None:
    save_result(compact_result, tmp_path)
    destination = plot_result(tmp_path, pattern_name="exp:1")
    assert destination == tmp_path / "overview.png"
    assert destination.stat().st_size > 1_000

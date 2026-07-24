"""Circular P1 finite-element domain construction."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skfem import Basis, ElementTriP1, MeshTri


@dataclass(frozen=True)
class Domain:
    """Triangular disk mesh and degree-of-freedom geometry in configured units."""

    mesh: MeshTri
    basis: Basis
    coordinates: NDArray[np.float64]
    triangles: NDArray[np.int64]
    boundary_dofs: NDArray[np.int64]
    interior_dofs: NDArray[np.int64]
    boundary_theta: NDArray[np.float64]
    h_max: float


def _maximum_edge_length(coordinates: NDArray[np.float64], triangles: NDArray[np.int64]) -> float:
    edges = np.concatenate(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        )
    )
    differences = coordinates[edges[:, 0]] - coordinates[edges[:, 1]]
    return float(np.max(np.linalg.norm(differences, axis=1)))


def build_disk(radius: float, refinements: int) -> Domain:
    """Build a uniformly refined radius-``radius`` disk with linear triangles."""

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"disk radius must be positive and finite; got {radius!r}")
    if isinstance(refinements, bool) or not isinstance(refinements, int) or refinements < 0:
        raise ValueError(f"refinements must be a nonnegative integer; got {refinements!r}")

    mesh = MeshTri.init_circle(nrefs=refinements, smoothed=True).scaled(radius)
    basis = Basis(mesh, ElementTriP1())
    coordinates = np.asarray(basis.doflocs.T, dtype=np.float64)
    triangles = np.asarray(mesh.t.T, dtype=np.int64)
    boundary_dofs = np.unique(np.asarray(basis.get_dofs().all(), dtype=np.int64))
    all_dofs = np.arange(basis.N, dtype=np.int64)
    interior_dofs = np.setdiff1d(all_dofs, boundary_dofs, assume_unique=True)
    theta = np.arctan2(
        coordinates[boundary_dofs, 1],
        coordinates[boundary_dofs, 0],
    ).astype(np.float64)

    return Domain(
        mesh=mesh,
        basis=basis,
        coordinates=coordinates,
        triangles=triangles,
        boundary_dofs=boundary_dofs,
        interior_dofs=interior_dofs,
        boundary_theta=theta,
        h_max=_maximum_edge_length(coordinates, triangles),
    )

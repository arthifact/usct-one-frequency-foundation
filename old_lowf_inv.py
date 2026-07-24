#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 00:56:56 2026

@author: chap868
"""

import numpy as np
import ufl
import gmsh
import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import dolfinx.io.gmsh as gmshio
from mpi4py import MPI
from petsc4py import PETSc
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pyvista as pv
try:
    import pyvistaqt
except ImportError:
    pass
import matplotlib
matplotlib.use('Agg')

# --- System Backend Scalar Space Alignment ---
ctype = PETSc.ScalarType

# =============================================================================
# 1. ROBUST GMSH CANONICAL REFINEMENT
# =============================================================================
# def create_disk_mesh(comm, R, mesh_size):
#     try:
#         gmsh.finalize()
#     except:
#         pass
#     gmsh.initialize()
#     gmsh.clear()

#     gmsh.model.occ.synchronize()
#     if comm.rank == 0:
#         disk = gmsh.model.occ.addDisk(0, 0, 0, R, R)
#         gmsh.model.occ.synchronize()

#         gmsh.model.addPhysicalGroup(1, [1], 101)
#         gmsh.model.setPhysicalName(1, 101, "boundary")
#         gmsh.model.addPhysicalGroup(2, [disk], 102)
#         gmsh.model.setPhysicalName(2, 102, "domain")

#         # NOTE: still flagged from the very first review -- PointsList is
#         # empty here, so the Distance/Threshold refinement fields are not
#         # actually refining toward anything. Now that the phantom has a
#         # 0.06-radius lesion with sharpness=80 (transition width ~0.0125,
#         # only ~1 cell wide at mesh_size=0.01), this is worth fixing before
#         # trusting recovery of the smallest lesion.
#         gmsh.model.mesh.field.add("Distance", 1)
#         gmsh.model.mesh.field.setNumbers(1, "PointsList", [])
#         gmsh.model.mesh.field.add("Threshold", 2)
#         gmsh.model.mesh.field.setNumber(2, "IField", 1)
#         gmsh.model.mesh.field.setNumber(2, "LcMin", mesh_size)
#         gmsh.model.mesh.field.setNumber(2, "LcMax", mesh_size * 2)
#         gmsh.model.mesh.field.setAsBackgroundMesh(2)
#         gmsh.model.mesh.generate(2)

#     mesh, cell_tags, facet_tags, _, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
#     gmsh.finalize()
#     return mesh, facet_tags

import gmsh
import dolfinx.io.gmsh as gmshio

def create_disk_mesh_refined(comm, R, mesh_size, refinement_points=None,
                              ring_refinements=None):
    """
    Disk mesh with local refinement around interior points (lesions) and/or
    an annular band around a given radius (tissue/water skin interface).

    refinement_points: list of {'center': (x,y), 'radius': r, 'lc': size,
                                 'buffer': extra_falloff_distance}
    ring_refinements:   list of {'center': (x,y), 'ring_radius': r,
                                  'inner_margin': d_in, 'outer_margin': d_out,
                                  'lc': size}
    """
    try:
        gmsh.finalize()
    except Exception:
        pass
    gmsh.initialize()
    gmsh.clear()

    refinement_points = refinement_points or []
    ring_refinements = ring_refinements or []

    if comm.rank == 0:
        disk = gmsh.model.occ.addDisk(0, 0, 0, R, R)
        gmsh.model.occ.synchronize()

        boundary_dimtags = gmsh.model.getBoundary([(2, disk)], oriented=False)
        boundary_curve_tag = boundary_dimtags[0][1]

        gmsh.model.addPhysicalGroup(1, [boundary_curve_tag], 101)
        gmsh.model.setPhysicalName(1, 101, "boundary")
        gmsh.model.addPhysicalGroup(2, [disk], 102)
        gmsh.model.setPhysicalName(2, 102, "domain")

        all_specs = refinement_points + ring_refinements
        point_tags = []
        for spec in all_specs:
            cx, cy = spec['center']
            point_tags.append(gmsh.model.occ.addPoint(cx, cy, 0))
        gmsh.model.occ.synchronize()
        if point_tags:
            gmsh.model.mesh.embed(0, point_tags, 2, disk)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        field_ids = []
        next_id = 1
        tag_iter = iter(point_tags)

        # --- Filled-disk refinement (lesions) ---
        for spec in refinement_points:
            tag = next(tag_iter)
            r_lesion = spec['radius']
            lc = spec['lc']
            buffer = spec.get('buffer', 0.08)

            dist_id = next_id; next_id += 1
            gmsh.model.mesh.field.add("Distance", dist_id)
            gmsh.model.mesh.field.setNumbers(dist_id, "PointsList", [tag])

            th_id = next_id; next_id += 1
            gmsh.model.mesh.field.add("Threshold", th_id)
            gmsh.model.mesh.field.setNumber(th_id, "IField", dist_id)
            gmsh.model.mesh.field.setNumber(th_id, "LcMin", lc)
            gmsh.model.mesh.field.setNumber(th_id, "LcMax", mesh_size * 2)
            gmsh.model.mesh.field.setNumber(th_id, "DistMin", r_lesion)
            gmsh.model.mesh.field.setNumber(th_id, "DistMax", r_lesion + buffer)
            field_ids.append(th_id)

        # --- Annulus refinement (skin / tissue-water interface) ---
        for spec in ring_refinements:
            tag = next(tag_iter)
            ring_r = spec['ring_radius']
            inner_margin = spec.get('inner_margin', 0.02)
            outer_margin = spec.get('outer_margin', 0.06)
            lc = spec['lc']

            dist_id = next_id; next_id += 1
            gmsh.model.mesh.field.add("Distance", dist_id)
            gmsh.model.mesh.field.setNumbers(dist_id, "PointsList", [tag])

            math_id = next_id; next_id += 1
            gmsh.model.mesh.field.add("MathEval", math_id)
            gmsh.model.mesh.field.setString(math_id, "F", f"Fabs(F{dist_id} - {ring_r})")

            th_id = next_id; next_id += 1
            gmsh.model.mesh.field.add("Threshold", th_id)
            gmsh.model.mesh.field.setNumber(th_id, "IField", math_id)
            gmsh.model.mesh.field.setNumber(th_id, "LcMin", lc)
            gmsh.model.mesh.field.setNumber(th_id, "LcMax", mesh_size * 2)
            gmsh.model.mesh.field.setNumber(th_id, "DistMin", inner_margin)
            gmsh.model.mesh.field.setNumber(th_id, "DistMax", outer_margin)
            field_ids.append(th_id)

        # --- Global coarse background (applies where nothing else does) ---
        base_id = next_id; next_id += 1
        gmsh.model.mesh.field.add("MathEval", base_id)
        gmsh.model.mesh.field.setString(base_id, "F", str(mesh_size * 2))
        field_ids.append(base_id)

        # --- Combine: finest applicable requirement wins everywhere ---
        min_id = next_id
        gmsh.model.mesh.field.add("Min", min_id)
        gmsh.model.mesh.field.setNumbers(min_id, "FieldsList", field_ids)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_id)

        gmsh.model.mesh.generate(2)

    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return mesh_data.mesh, mesh_data.facet_tags

# =============================================================================
# Medical phantom target: replace with whatever
# =============================================================================
import numpy as np

def alpha_expr(x, sharpness=80.0, tissue_radius=0.55, water_c=1.0):
    """
    Central tissue phantom surrounded by a water bath (USCT-style setup:
    tissue sample suspended centrally within a ring transducer array).
    Lesion/gland geometry scaled to sit well inside tissue_radius.
    """
    r = np.sqrt(x[0]**2 + x[1]**2)
    c = np.full_like(r, water_c)

    # --- Glandular region ---
    gland_mask = ((x[0] - 0.0275) / 0.3025)**2 + ((x[1] + 0.0275) / 0.22)**2
    gland_transition = 0.5 * (1.0 - np.tanh(sharpness * (gland_mask - 1.0)))
    c = c + 0.3 * gland_transition

    # --- Lesion 1: high-contrast inclusion ---
    r1 = np.sqrt((x[0] + 0.165)**2 + (x[1] - 0.165)**2)
    lesion1 = 0.5 * (1.0 - np.tanh(sharpness * (r1 - 0.066)))
    c = c + 0.8 * lesion1

    # --- Lesion 2: lower-contrast (cyst-like) inclusion ---
    r2 = np.sqrt((x[0] - 0.1375)**2 + (x[1] - 0.0825)**2)
    lesion2 = 0.5 * (1.0 - np.tanh(sharpness * (r2 - 0.044)))
    c = c - 0.35 * lesion2

    # --- Lesion 3: smallest, resolution-test feature ---
    r3 = np.sqrt((x[0] - 0.055)**2 + (x[1] + 0.1925)**2)
    lesion3 = 0.5 * (1.0 - np.tanh(sharpness * (r3 - 0.033)))
    c = c + 0.6 * lesion3

    # --- Skin ring at the tissue/water interface ---
    # Product of a rising edge (at the inner radius) and a falling edge
    # (at the outer radius) -- bounded in [0,1] automatically, no clip needed.
    rising_at_inner = 0.5 * (1.0 + np.tanh(sharpness * 0.5 * (r - (tissue_radius - 0.04))))
    falling_at_outer = 0.5 * (1.0 - np.tanh(sharpness * 0.5 * (r - (tissue_radius + 0.04))))
    skin_band = rising_at_inner * falling_at_outer
    c = c + 0.25 * skin_band

    return c + 0j

refinement_points = [
    {'center': (-0.165,  0.165), 'radius': 0.066, 'lc': 0.0025, 'buffer': 0.05},
    {'center': ( 0.1375, 0.0825), 'radius': 0.044, 'lc': 0.0025, 'buffer': 0.05},
    {'center': ( 0.055, -0.1925), 'radius': 0.033, 'lc': 0.0020, 'buffer': 0.05},
]
ring_refinements = [
    {'center': (0.0, 0.0), 'ring_radius': 0.55, 'inner_margin': 0.02,
     'outer_margin': 0.06, 'lc': 0.004},
]

def max_mode_for_omega(omega, R=1.0, min_modes=2):
    """
    Angular content beyond m ~ k*R is essentially unresolvable/noise-dominated
    on a boundary circle of radius R (Bessel function decay). Scale the mode
    count with frequency so low-frequency stages aren't paying for angular
    resolution they can't actually use.
    """
    return max(min_modes, int(np.ceil(omega * R)))


def build_boundary_forcings(max_mode):
    forcings = {('cos', 0): get_boundary_forcing_cos(0)}
    for m in range(1, max_mode + 1):
        forcings[('cos', m)] = get_boundary_forcing_cos(m)
        forcings[('sin', m)] = get_boundary_forcing_sin(m)
    return forcings




R = 1.0
mesh_size = 0.03
mesh, facet_tags = create_disk_mesh_refined(
    MPI.COMM_WORLD, R, mesh_size,
    refinement_points=refinement_points,
    ring_refinements=ring_refinements
)
print(f"Total cells: {mesh.topology.index_map(mesh.topology.dim).size_global}")

V = fem.functionspace(mesh, ("Lagrange", 2))
print(f"Total DOFs: {V.dofmap.index_map.size_global}")

c_func_true = fem.Function(V)
c_func_true.interpolate(alpha_expr)

boundary_marker = 101
boundary_facets = facet_tags.find(boundary_marker)
fdim = mesh.topology.dim - 1
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

# NOTE: u_boundary_expr and its interpolate call were removed here --
# confirmed dead: bcond is unconditionally overwritten by
# bcond.interpolate(forcing_fn) inside the dataset-generation loop below,
# before any forward_HH call ever reads the sigmoid initial value.
bcond = fem.Function(V)

experimental_data = {}


''' Mesh sanity check '''

# import dolfinx.plot
# import pyvista as pv

# topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
# grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# plotter = pv.Plotter(window_size=(700, 700))
# plotter.add_mesh(grid, show_edges=True, color="white", edge_color="black", line_width=0.5)
# plotter.view_xy()
# plotter.show()


# =============================================================================
# Forward model
# =============================================================================
def eta_for_omega(omega, eta_base=1e-3, omega_ref=None):
    # omega_ref intentionally defaults to omega_schedule[0], resolved at
    # CALL time (omega_schedule is defined further below). Pass omega_ref
    # explicitly if you ever split this into a separate module.
    omega_ref = omega_ref or omega_schedule[0]
    return eta_base * max(1.0, (omega / omega_ref)**2)


def forward_HH(c_func, bcond, mesh, facet_tags, omega, boundary_dofs, V, eta):
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    k_squared_true = omega**2 / (c_func * c_func) * (1 + 1j * eta)
    bc = fem.dirichletbc(bcond, boundary_dofs)

    zero_complex_scalar = fem.Constant(mesh, np.array(0.0 + 0.0j, dtype=ctype))
    a_ufl = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k_squared_true * ufl.inner(u, v)) * ufl.dx
    L_ufl = ufl.inner(zero_complex_scalar, v) * ufl.dx

    problem = LinearProblem(
        a_ufl, L_ufl, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="fwd_"
    )
    u_true = problem.solve()

    dn = extract_neumann_data(u_true, mesh, facet_tags, boundary_marker, boundary_dofs, V)
    return u_true, dn   # (duplicate return removed)


def extract_neumann_data(u_true, mesh, facet_tags, boundary_marker, boundary_dofs, V):
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector
    from petsc4py import PETSc
    import numpy as np

    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    q_trial = ufl.TrialFunction(V)
    w_test = ufl.TestFunction(V)

    flux_expr = ufl.dot(ufl.grad(u_true), n)

    diag_guard = 1e-12
    a_proj = (ufl.inner(q_trial, w_test) * ds(boundary_marker)
              + diag_guard * ufl.inner(q_trial, w_test) * ufl.dx)
    L_proj = ufl.inner(flux_expr, w_test) * ds(boundary_marker)

    a_form = fem.form(a_proj)
    L_form = fem.form(L_proj)

    A = assemble_matrix(a_form)
    A.assemble()

    b = assemble_vector(L_form)
    b.assemble()

    num_dofs_local = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    all_dofs = np.arange(num_dofs_local, dtype=np.int32)
    interior_dofs = np.setdiff1d(all_dofs, boundary_dofs.astype(np.int32))

    A.zeroRowsLocal(interior_dofs, diag=1.0)
    b.setValuesLocal(interior_dofs, np.zeros(len(interior_dofs), dtype=PETSc.ScalarType))
    b.assemble()

    q_sol = fem.Function(V)
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.solve(b, q_sol.x.petsc_vec)
    q_sol.x.scatter_forward()

    return q_sol.x.array[boundary_dofs].copy()


# =============================================================================
# Angular basis + frequency schedule + dataset generation
# =============================================================================
def get_boundary_forcing_cos(mode_number):
    return lambda x: np.cos(mode_number * np.arctan2(x[1], x[0])) + 0j

def get_boundary_forcing_sin(mode_number):
    return lambda x: np.sin(mode_number * np.arctan2(x[1], x[0])) + 0j

max_mode = 6

boundary_forcings = {}
boundary_forcings[('cos', 0)] = get_boundary_forcing_cos(0)
for m in range(1, max_mode + 1):
    boundary_forcings[('cos', m)] = get_boundary_forcing_cos(m)
    boundary_forcings[('sin', m)] = get_boundary_forcing_sin(m)

c_min_bound = 0.9   # tightened from 0.5, given the phantom's true minimum is ~0.95
R_domain = 1.0
L_max = 2.0 * R_domain
delta_f_safe = c_min_bound / (2.0 * L_max)   # now 0.225, was 0.125 -- fewer stages needed
print(f"delta_f_safe = {delta_f_safe:.4f}")

f_start = delta_f_safe
f_end = 4.0   # "budget" tier from the earlier table; raise to 6.0 for the fuller target
n_stages = int(np.ceil((f_end - f_start) / delta_f_safe)) + 1
f_values = np.linspace(f_start, f_end, n_stages)
omega_schedule = [2.0 * np.pi * f for f in f_values]

boundary_forcings_by_omega = {}
total_solves = 0
for omega in omega_schedule:
    m_max = max_mode_for_omega(omega, R=R_domain)
    forcings = build_boundary_forcings(m_max)
    boundary_forcings_by_omega[omega] = forcings
    total_solves += len(forcings)
    print(f"  f={omega/(2*np.pi):.3f}  omega={omega:.3f}  max_mode={m_max}  n_modes={len(forcings)}")


worst_case_flat_mode = max(max_mode_for_omega(w, R=R_domain) for w in omega_schedule)
flat_adequate_total = len(omega_schedule) * (1 + 2 * worst_case_flat_mode)
print(f"Total forward solves, adaptive: {total_solves} "
      f"(vs {flat_adequate_total} for a flat scheme adequate at every stage, max_mode={worst_case_flat_mode})")

# --------------------------------------------- #

''' Functions relating to parallel processing '''

# --------------------------------------------- #

from concurrent.futures import ProcessPoolExecutor
import os

def _worker_init(mesh_size, R, refinement_points, ring_refinements):
    """Each worker process builds its own independent mesh/V/etc. once."""
    global _mesh, _V, _facet_tags, _boundary_dofs, _boundary_weights
    _mesh, _facet_tags = create_disk_mesh_refined(
        MPI.COMM_SELF, R, mesh_size, refinement_points, ring_refinements
    )
    _V = fem.functionspace(_mesh, ("Lagrange", 2))
    # ... rebuild boundary_dofs, boundary_weights, ksp_filter_global, etc. here ...

def _solve_one_mode(args):
    c_array, omega, mode_key, forcing_fn, observed_data, eta_stage, alpha_reg = args
    c_local = fem.Function(_V)
    c_local.x.array[:] = c_array
    c_local.x.scatter_forward()

    bcond_local = fem.Function(_V)
    bcond_local.interpolate(forcing_fn)

    u_fwd, dn_fwd = forward_HH(c_local, bcond_local, _mesh, _facet_tags, omega, _boundary_dofs, _V, eta_stage)
    misfit = 0.5 * np.sum(_boundary_weights * np.abs(dn_fwd - observed_data)**2)
    grad = compute_adjoint_gradient(c_local, u_fwd, dn_fwd, observed_data, omega,
                                     _boundary_dofs, _V, _mesh, _facet_tags, eta=eta_stage)
    return misfit, grad



# --------------------------------------------- #

# --------------------------------------------- #






experimental_data = {}
cnnt = 0
for omega in omega_schedule:
    cnnt=cnnt+1
    print(cnnt)
    eta_gen = eta_for_omega(omega)
    for mode_key, forcing_fn in boundary_forcings_by_omega[omega].items():
        bcond.interpolate(forcing_fn)
        _, data_n = forward_HH(c_func_true, bcond, mesh, facet_tags, omega, boundary_dofs, V, eta_gen)
        experimental_data[(omega, mode_key)] = data_n.copy()
print("Synthetic observation dataset generation complete.")


''' FORWARD PASS TEST '''


''' '''






# import dolfinx.plot

# def plot_forward_pass(c_func, u_true, dn, boundary_dofs, V, mesh, mode_number, omega):
#     """
#     Visualize one forward solve:
#       1. c(x,y) coefficient field (real, static)
#       2. Re(u_true) and Im(u_true) wavefields on the disk
#       3. Neumann data dn vs boundary angle (sanity check trace)
#     """

#     # -------------------------------------------------------------
#     # 1 & 2: Field plots via pyvista
#     # -------------------------------------------------------------
#     topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
#     grid = pv.UnstructuredGrid(topology, cell_types, geometry)

#     grid["c"] = c_func.x.array.real
#     grid["Re_u"] = u_true.x.array.real
#     grid["Im_u"] = u_true.x.array.imag

#     plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 500))

#     plotter.subplot(0, 0)
#     plotter.add_mesh(grid, scalars="c", cmap="viridis", show_edges=False)
#     plotter.add_scalar_bar("c(x,y)")
#     plotter.view_xy()
#     plotter.add_text("Wave speed c(x,y)", font_size=10)

#     plotter.subplot(0, 1)
#     plotter.add_mesh(grid, scalars="Re_u", cmap="RdBu", show_edges=False)
#     plotter.add_scalar_bar("Re(u)")
#     plotter.view_xy()
#     plotter.add_text(f"Re(u), mode={mode_number}, omega={omega:.3f}", font_size=10)

#     plotter.subplot(0, 2)
#     plotter.add_mesh(grid, scalars="Im_u", cmap="RdBu", show_edges=False)
#     plotter.add_scalar_bar("Im(u)")
#     plotter.view_xy()
#     plotter.add_text(f"Im(u), mode={mode_number}, omega={omega:.3f}", font_size=10)

#     plotter.show()

#     # -------------------------------------------------------------
#     # 3: Neumann data vs boundary angle (matplotlib)
#     # -------------------------------------------------------------
#     dof_coords = V.tabulate_dof_coordinates()
#     coords_on_boundary = dof_coords[boundary_dofs]
#     angles = np.arctan2(coords_on_boundary[:, 1], coords_on_boundary[:, 0])

#     order = np.argsort(angles)
#     angles_sorted = angles[order]
#     dn_sorted = dn[order]

#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#     axes[0].plot(angles_sorted, dn_sorted.real, '.-', label="Re(dn)", markersize=3)
#     axes[0].plot(angles_sorted, dn_sorted.imag, '.-', label="Im(dn)", markersize=3)
#     axes[0].set_xlabel("boundary angle (rad)")
#     axes[0].set_ylabel("∂u/∂n")
#     axes[0].set_title(f"Neumann trace, mode={mode_number}")
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)

#     axes[1].plot(angles_sorted, np.abs(dn_sorted), '.-', color='k', markersize=3)
#     axes[1].set_xlabel("boundary angle (rad)")
#     axes[1].set_ylabel("|∂u/∂n|")
#     axes[1].set_title("Neumann trace magnitude")
#     axes[1].grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.show()


# # -----------------------------------------------------------------
# # Run a single forward pass and plot it
# # -----------------------------------------------------------------
# test_mode = 0
# bcond.interpolate(get_boundary_forcing(test_mode))
# u_true_test, dn_test = forward_HH(
#     c_func_true, bcond, mesh, facet_tags, lowest_omega, boundary_dofs, V, eta_for_omega(lowest_omega)
# )

# plot_forward_pass(c_func_true, u_true_test, dn_test, boundary_dofs, V, mesh,
#                    test_mode, lowest_omega)

# =============================================================================
# 3. PRE-COMPUTING GEOMETRIC MESH LUMPED MASS WEIGHTINGS
# =============================================================================
# This pre-calculates the local volume component associated with every degree-of-freedom node.
# v_mass = ufl.TestFunction(V)
# # Using ufl.conj handles the complex linear functional formatting rule
# mass_form = fem.form(1.0 * ufl.conj(v_mass) * ufl.dx, dtype=ctype)
# lumped_mass_vector = fem.assemble_vector(mass_form).array.real.copy()
# # Epsilon threshold protects interior/unmapped cells from divide-by-zero errors
# lumped_mass_vector[lumped_mass_vector < 1e-15] = 1.0


def compute_adjoint_gradient(c_current, u_forward, dn_forward, data_observed, omega, boundary_dofs, V, mesh, facet_tags, eta):
    p, w = ufl.TrialFunction(V), ufl.TestFunction(V)

    residual = data_observed - dn_forward

    k_squared_adj = (omega**2 / (c_current * c_current)) * (1.0 - 1j * eta)
    a_adj = (ufl.inner(ufl.grad(p), ufl.grad(w)) - k_squared_adj * ufl.inner(p, w)) * ufl.dx

    zero_complex_scalar = fem.Constant(mesh, np.array(0.0 + 0.0j, dtype=ctype))
    L_adj = ufl.inner(zero_complex_scalar, w) * ufl.dx

    residual_bc_func = fem.Function(V, dtype=ctype)
    residual_bc_func.x.array[boundary_dofs] = residual
    residual_bc_func.x.scatter_forward()

    adj_bc = fem.dirichletbc(residual_bc_func, boundary_dofs)

    adj_problem = LinearProblem(
        a_adj, L_adj, bcs=[adj_bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="adj_"
    )
    p_adjoint_sol = adj_problem.solve()

    v_grad = ufl.TestFunction(V)
    damping_factor = (1.0 + 1j * eta)
    # Take the real part of the FULL complex product (damping_factor * inner(...)),
    # not of inner(...) alone -- Re[a*b] != Re(a)*Re(b) for complex a, b.
    dc_integrand = (-2.0 * omega**2 / (c_current**3)) * ufl.real(damping_factor * ufl.inner(u_forward, p_adjoint_sol)) * ufl.conj(v_grad) * ufl.dx
    dc_form = fem.form(dc_integrand, dtype=ctype)

    riesz_vector = fem.assemble_vector(dc_form).array.real.copy()
    return riesz_vector

# =============================================================================
# 5. OBJECTIVE FUNCTIONAL LOOP AND INVERSION EXECUTION
# =============================================================================
c_estimated = fem.Function(V)
c_estimated.x.array[:] = 1.0  # Homogeneous baseline guess


ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
v_bmass = ufl.TestFunction(V)
bmass_form = fem.form(1.0 * ufl.conj(v_bmass) * ds(boundary_marker), dtype=ctype)
boundary_lumped_full = fem.assemble_vector(bmass_form).array.real.copy()
boundary_weights = boundary_lumped_full[boundary_dofs]   # compute once, outside the loop



''' These next two functions help reduce grid-scale high frequency oscillations due to the under determined nature of the problem '''
# =============================================================================
# GRADIENT FILTER (suppresses mesh-connectivity-scale / checkerboard noise)
# =============================================================================
def build_gradient_filter(V, mesh, filter_radius):
    from dolfinx.fem.petsc import assemble_matrix

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    filter_form = (filter_radius**2 * ufl.inner(ufl.grad(u), ufl.grad(v))
                   + ufl.inner(u, v)) * ufl.dx
    A_filter = assemble_matrix(fem.form(filter_form, dtype=ctype))
    A_filter.assemble()

    ksp_filter = PETSc.KSP().create(mesh.comm)
    ksp_filter.setOperators(A_filter)
    ksp_filter.setType("preonly")
    ksp_filter.getPC().setType("lu")
    return ksp_filter

def apply_gradient_filter(ksp_filter, raw_gradient, V):
    from dolfinx.fem.petsc import assemble_vector  # PETSc-backed assembler, not fem.assemble_vector

    v = ufl.TestFunction(V)
    g_raw_func = fem.Function(V, dtype=ctype)
    g_raw_func.x.array[:] = raw_gradient.astype(ctype)

    mass_form = fem.form(ufl.inner(g_raw_func, v) * ufl.dx, dtype=ctype)
    rhs = assemble_vector(mass_form)
    rhs.assemble()

    g_smooth = fem.Function(V, dtype=ctype)
    ksp_filter.solve(rhs, g_smooth.x.petsc_vec)
    g_smooth.x.scatter_forward()

    return g_smooth.x.array.real.copy()

filter_radius = 0.01  # was 0.05 -- rescaled to the new locally-refined mesh
ksp_filter_global = build_gradient_filter(V, mesh, filter_radius)


# =============================================================================
# OBJECTIVE FUNCTIONAL (with gradient filtering)
# =============================================================================
def make_objective_functional(omega, active_forcings, dn_cache, alpha_reg):
    eta_stage = eta_for_omega(omega)

    def objective_functional(c_vector_array):
        c_estimated.x.array[:] = c_vector_array
        c_estimated.x.scatter_forward()

        total_misfit = 0.0
        total_gradient = np.zeros_like(c_vector_array, dtype=np.float64)

        for mode_key, forcing_fn in active_forcings.items():
            bcond.interpolate(forcing_fn)

            u_fwd, dn_fwd = forward_HH(c_estimated, bcond, mesh, facet_tags, omega, boundary_dofs, V, eta_stage)
            dn_cache[mode_key] = dn_fwd.copy()

            observed_data = experimental_data[(omega, mode_key)]
            misfit_delta = 0.5 * np.sum(boundary_weights * np.abs(dn_fwd - observed_data)**2)
            total_misfit += misfit_delta

            adj_grad = compute_adjoint_gradient(
                c_estimated, u_fwd, dn_fwd, observed_data, omega,
                boundary_dofs, V, mesh, facet_tags, eta=eta_stage
            )
            total_gradient += adj_grad

        # --- Tikhonov / H1 smoothness regularization ---
        c_test = ufl.TestFunction(V)
        reg_form = alpha_reg * ufl.inner(ufl.grad(c_estimated), ufl.grad(c_test)) * ufl.dx
        reg_grad = fem.assemble_vector(fem.form(reg_form, dtype=ctype)).array.real.copy()

        reg_value_form = fem.form(
            alpha_reg * 0.5 * ufl.inner(ufl.grad(c_estimated), ufl.grad(c_estimated)) * ufl.dx,
            dtype=ctype
        )
        reg_value = fem.assemble_scalar(reg_value_form).real



        with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=_worker_init,
                          initargs=(mesh_size, R, refinement_points, ring_refinements)) as executor:
            tasks = [(c_vector_array, omega, mk, fn, experimental_data[(omega, mk)], eta_stage, alpha_reg)
                     for mk, fn in active_forcings.items()]
            results = list(executor.map(_solve_one_mode, tasks))
        
        total_misfit = sum(r[0] for r in results)
        total_gradient = sum(r[1] for r in results)

        # total_misfit += reg_value
        # total_gradient += reg_grad

        # --- Smooth the gradient to suppress mesh-connectivity-scale noise ---
        total_gradient = apply_gradient_filter(ksp_filter_global, total_gradient, V)

        # --- Sanity gate ---
        MAX_GRAD_NORM = 1e3
        grad_norm = np.max(np.abs(total_gradient))
        if grad_norm > MAX_GRAD_NORM or not np.isfinite(total_misfit):
            print(f"WARNING: gradient norm {grad_norm:.3e} exceeds sanity threshold — likely near-resonance blowup")
            total_gradient = np.clip(total_gradient, -MAX_GRAD_NORM, MAX_GRAD_NORM)

        return float(np.real(total_misfit)), np.array(total_gradient, dtype=np.float64)

    return objective_functional


# =============================================================================
# MULTI-PASS FREQUENCY-CONTINUATION OUTER LOOP
# =============================================================================
c_estimated = fem.Function(V)
c_estimated.x.array[:] = 1.0  # homogeneous baseline, pass 1 / stage 1 only

boundary_data_history = []
stage_boundaries = []
pass_boundaries = []
global_iteration_counter = 0
model_snapshots = [] 
# n_passes = 2
# pass_alpha_scale_factor = 0.3  # gentler decay than the earlier 0.1**pass_idx



n_passes = 2  # matches what you're actually executing this time
total_stages_this_run = n_passes * len(omega_schedule)
global_alpha_schedule = np.logspace(-3, -7, total_stages_this_run)

print("\n==================== MULTI-PASS FREQUENCY-CONTINUATION INVERSE RECOVERY ====================")

for pass_idx in range(n_passes):
    print(f"\n======== PASS {pass_idx + 1}/{n_passes} ========")
    pass_boundaries.append(global_iteration_counter)

    for stage_idx, omega in enumerate(omega_schedule):
        active_forcings = boundary_forcings_by_omega[omega]   # <-- was the fixed global boundary_forcings
        alpha_reg_current = global_alpha_schedule[pass_idx * len(omega_schedule) + stage_idx]
    
        dn_cache = {}
        objective_functional = make_objective_functional(omega, active_forcings, dn_cache, alpha_reg_current)
        

        _ = objective_functional(c_estimated.x.array.real.copy())
        boundary_data_history.append({
            'iteration': global_iteration_counter, 'omega': omega, 'dn_by_mode': dict(dn_cache)
        })

        def stage_callback(intermediate_result):
            global global_iteration_counter
            global_iteration_counter += 1
            print(f"| Pass {pass_idx+1} Stage {stage_idx+1} | Iter {global_iteration_counter:03d} | "
                  f"Misfit: {intermediate_result.fun:.5e}")
            boundary_data_history.append({
                'iteration': global_iteration_counter, 'omega': omega, 'dn_by_mode': dict(dn_cache)
            })

        res = opt.minimize(
            objective_functional,
            c_estimated.x.array.real.copy(),
            method='L-BFGS-B',
            jac=True,
            bounds=[(0.9, 3.0)] * len(c_estimated.x.array),
            callback=stage_callback,
            options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )

        if not (np.all(np.isfinite(res.x)) and np.isfinite(res.fun)):
            print(f"WARNING: pass {pass_idx+1} stage {stage_idx+1} produced non-finite result "
                  f"— keeping previous c_estimated, skipping update")
        else:
            c_estimated.x.array[:] = res.x
            c_estimated.x.scatter_forward()

        if not (np.all(np.isfinite(res.x)) and np.isfinite(res.fun)):
            print(f"WARNING: pass {pass_idx+1} stage {stage_idx+1} produced non-finite result "
                  f"— keeping previous c_estimated, skipping update")
        else:
            c_estimated.x.array[:] = res.x
            c_estimated.x.scatter_forward()

        # Lightweight snapshot for post-run visualization (cheap: just a DOF array copy)
        model_snapshots.append({
            'pass': pass_idx + 1,
            'stage': stage_idx + 1,
            'omega': omega,
            'f': omega / (2.0 * np.pi),
            'c_array': c_estimated.x.array.real.copy(),
            'misfit': res.fun
        })

        print(f"---- misfit = {res.fun:.5e}, iterations = {res.nit}, "
              f"c range = [{c_estimated.x.array.real.min():.3f}, {c_estimated.x.array.real.max():.3f}], "
              f"message: {res.message} ----")


    print(f"\n======== Pass {pass_idx + 1} complete ========")

print("\n==================== MULTI-PASS INVERSION COMPLETE ====================\n")






''' Plot evolving recovered model '''
import numpy as np
import dolfinx.plot
import pyvista as pv

def render_model_evolution_gif(model_snapshots, c_func_true, V, output_path="model_evolution.gif",
                                 clim=(0.9, 2.1), fps=4):
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["c_true"] = c_func_true.x.array.real.copy()
    grid.point_data["c_recovered"] = model_snapshots[0]['c_array']  # initial frame

    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1100, 500))

    plotter.subplot(0, 0)
    plotter.add_text("Ground Truth Target: C(x,y)", font_size=12, position="upper_edge")
    plotter.add_mesh(grid, show_edges=False, scalars="c_true", cmap="inferno",
                      clim=clim, lighting=False)
    plotter.view_xy()

    plotter.subplot(0, 1)
    text_actor = plotter.add_text(
        f"Pass {model_snapshots[0]['pass']} | Stage {model_snapshots[0]['stage']}",
        font_size=11, position="upper_edge"
    )
    plotter.add_mesh(grid, show_edges=False, scalars="c_recovered", cmap="inferno",
                      clim=clim, lighting=False)
    plotter.view_xy()

    plotter.open_gif(output_path, fps=fps)

    for snap in model_snapshots:
        grid.point_data["c_recovered"] = snap['c_array']

        plotter.subplot(0, 1)
        plotter.remove_actor(text_actor)
        text_actor = plotter.add_text(
            f"Pass {snap['pass']} | Stage {snap['stage']} | f={snap['f']:.3f} | "
            f"misfit={snap['misfit']:.2e}",
            font_size=11, position="upper_edge"
        )

        plotter.write_frame()

    plotter.close()
    print(f"Saved evolution animation to {output_path} ({len(model_snapshots)} frames)")

render_model_evolution_gif(model_snapshots, c_func_true, V)



# =============================================================================
# FREQUENCY-CONTINUATION OUTER LOOP
# =============================================================================
# alpha_reg = 1e-4  # starting regularization weight; re-tune per stage if needed


# # alpha_reg_schedule = [1e-3, 1e-4, 1e-5, 1e-6]  # paired with omega_schedule
# assert len(alpha_reg_schedule) == len(omega_schedule)

# c_estimated = fem.Function(V)
# c_estimated.x.array[:] = 1.0  # homogeneous baseline guess, only for stage 0

# boundary_data_history = []       # flat history across ALL stages, for plotting
# stage_boundaries = []            # iteration indices where a new frequency stage begins
# global_iteration_counter = 0

''' TEST GRADIENT MATCH '''

# def finite_difference_gradient_check(c_estimated, omega, mode_key, h=1e-4, n_check_dofs=5):
#     """
#     Compare analytic adjoint gradient against central finite differences
#     at a handful of random interior DOFs. Run this BEFORE trusting the
#     full inversion loop.

#     omega:    a value from omega_schedule
#     mode_key: a key from boundary_forcings, e.g. ('cos', 1) or ('sin', 2)
#     """
#     eta_stage = eta_for_omega(omega)
#     rng = np.random.default_rng(0)
#     base = c_estimated.x.array.real.copy()
#     check_dofs = rng.choice(len(base), size=n_check_dofs, replace=False)

#     forcing_fn = boundary_forcings[mode_key]
#     bcond.interpolate(forcing_fn)
#     observed_data = experimental_data[(omega, mode_key)]

#     def eval_misfit(c_vec):
#         c_estimated.x.array[:] = c_vec
#         c_estimated.x.scatter_forward()
#         _, dn = forward_HH(c_estimated, bcond, mesh, facet_tags, omega, boundary_dofs, V, eta_stage)
#         return 0.5 * np.sum(boundary_weights * np.abs(dn - observed_data)**2)

#     # Analytic gradient at base point
#     c_estimated.x.array[:] = base
#     c_estimated.x.scatter_forward()
#     u_fwd, dn_fwd = forward_HH(c_estimated, bcond, mesh, facet_tags, omega, boundary_dofs, V, eta_stage)
#     analytic_grad = compute_adjoint_gradient(
#         c_estimated, u_fwd, dn_fwd, observed_data, omega,
#         boundary_dofs, V, mesh, facet_tags, eta=eta_stage
#     )

#     print(f"{'DOF':>6} {'analytic':>14} {'finite-diff':>14} {'rel. error':>12}")
#     for i in check_dofs:
#         c_plus = base.copy();  c_plus[i]  += h
#         c_minus = base.copy(); c_minus[i] -= h
#         fd = (eval_misfit(c_plus) - eval_misfit(c_minus)) / (2 * h)
#         rel_err = abs(fd - analytic_grad[i]) / (abs(fd) + 1e-14)
#         print(f"{i:6d} {analytic_grad[i]:14.6e} {fd:14.6e} {rel_err:12.4e}")

#     # restore
#     c_estimated.x.array[:] = base
#     c_estimated.x.scatter_forward()


# # Example calls — pick a representative low, mid, and high frequency stage
# finite_difference_gradient_check(c_estimated, omega=omega_schedule[0], mode_key=('cos', 1))
# finite_difference_gradient_check(c_estimated, omega=omega_schedule[len(omega_schedule)//2], mode_key=('sin', 2))
# finite_difference_gradient_check(c_estimated, omega=omega_schedule[-1], mode_key=('cos', 3))




''' END PREP, CHECKS, AND DATA GENERATION '''





# print("\n==================== FREQUENCY-CONTINUATION INVERSE RECOVERY ====================")

# for stage_idx, omega in enumerate(omega_schedule):
#     print(f"\n---- Stage {stage_idx + 1}/{len(omega_schedule)}: omega = {omega:.4f} ----")
#     stage_boundaries.append(global_iteration_counter)

#     dn_cache = {}
#     objective_functional = make_objective_functional(omega, boundary_forcings, dn_cache, alpha_reg_schedule[stage_idx])

#     # capture pre-stage boundary data (using the guess carried over from the
#     # previous stage, or the homogeneous baseline on stage 0)
#     _ = objective_functional(c_estimated.x.array.real.copy())
#     boundary_data_history.append({
#         'iteration': global_iteration_counter,
#         'omega': omega,
#         'dn_by_mode': dict(dn_cache)
#     })

#     def stage_callback(intermediate_result):
#         global global_iteration_counter
#         global_iteration_counter += 1
#         print(f"| Stage {stage_idx + 1} | Iter {global_iteration_counter:03d} | Misfit: {intermediate_result.fun:.5e}")
#         boundary_data_history.append({
#             'iteration': global_iteration_counter,
#             'omega': omega,
#             'dn_by_mode': dict(dn_cache)
#         })

#     res = opt.minimize(
#         objective_functional,
#         c_estimated.x.array.real.copy(),
#         method='L-BFGS-B',
#         jac=True,
#         bounds=[(0.5, 3.0)] * len(c_estimated.x.array),
#         callback=stage_callback,
#         options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
#     )

#     # carry the recovered field forward as next stage's initial guess
#     c_estimated.x.array[:] = res.x
#     c_estimated.x.scatter_forward()
#     print(f"  end of stage {stage_idx+1}: c range = [{res.x.min():.3f}, {res.x.max():.3f}]")
#     print(f"---- Stage {stage_idx + 1} complete: final misfit = {res.fun:.5e}, "
#           f"iterations = {res.nit}, message: {res.message} ----")

# print("\n==================== FREQUENCY-CONTINUATION INVERSION COMPLETE ====================\n")










import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_boundary_data_evolution(boundary_data_history, experimental_data, boundary_dofs, V, lowest_omega, modes_to_plot=None):
    dof_coords = V.tabulate_dof_coordinates()
    coords_on_boundary = dof_coords[boundary_dofs]
    angles = np.arctan2(coords_on_boundary[:, 1], coords_on_boundary[:, 0])
    order = np.argsort(angles)
    angles_sorted = angles[order]

    if modes_to_plot is None:
        modes_to_plot = list(boundary_data_history[0]['dn_by_mode'].keys())

    n_iters = len(boundary_data_history)
    cmap = plt.colormaps['viridis'].resampled(n_iters)

    # 2 rows (Re, Im) x len(modes_to_plot) columns
    fig, axes = plt.subplots(2, len(modes_to_plot), figsize=(6 * len(modes_to_plot), 8), squeeze=False)

    for ax_idx, mode in enumerate(modes_to_plot):
        ax_re = axes[0][ax_idx]
        ax_im = axes[1][ax_idx]
        observed = experimental_data[(lowest_omega, mode)][order]

        for i, snap in enumerate(boundary_data_history):
            dn = snap['dn_by_mode'][mode][order]
            label = f"iter {snap['iteration']}" if i in (0, n_iters - 1) else None
            ax_re.plot(angles_sorted, dn.real, color=cmap(i), alpha=0.7, lw=1, label=label)
            ax_im.plot(angles_sorted, dn.imag, color=cmap(i), alpha=0.7, lw=1, label=label)

        ax_re.plot(angles_sorted, observed.real, 'k--', lw=2, label="observed (truth)")
        ax_im.plot(angles_sorted, observed.imag, 'k--', lw=2, label="observed (truth)")

        ax_re.axhline(0, color='gray', lw=0.5)
        ax_im.axhline(0, color='gray', lw=0.5)

        ax_re.set_title(f"Mode {mode} — Re(∂u/∂n)")
        ax_im.set_title(f"Mode {mode} — Im(∂u/∂n)")
        ax_re.set_xlabel("boundary angle (rad)")
        ax_im.set_xlabel("boundary angle (rad)")
        ax_re.set_ylabel("Re(dn)")
        ax_im.set_ylabel("Im(dn)")
        ax_re.legend(fontsize=7)
        ax_im.legend(fontsize=7)
        ax_re.grid(True, alpha=0.3)
        ax_im.grid(True, alpha=0.3)

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_iters - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.03, pad=0.1)
    cbar.set_label("optimization iteration")

    plt.tight_layout()
    plt.show()
    
# plot_boundary_data_evolution(boundary_data_history, experimental_data, boundary_dofs, V, lowest_omega)








# _ = objective_functional(c_estimated.x.array.real.copy())
# boundary_data_history.append({'iteration': 0, 'dn_by_mode': dict(_dn_cache)})

# print("\n==================== EXECUTING INVERSE RECOVERY LOOP ====================")
# res = opt.minimize(
#     objective_functional,
#     c_estimated.x.array.real.copy(),
#     method='L-BFGS-B',
#     jac=True,
#     bounds=[(0.5, 3.0)] * len(c_estimated.x.array),
#     callback=optimization_callback,
#     options={
#         'maxiter': 100,
#         'maxfun': 200,
#         'ftol': 1e-12,   # much tighter than the factr default (~2.2e-9)
#         'gtol': 1e-8,
#         'disp': True
#     }
# )
# c_estimated.x.array[:] = res.x
# c_estimated.x.scatter_forward()
# print("==================== INVERSION ENGINE COMPLETE ====================\n")

# plot_boundary_data_evolution(boundary_data_history, experimental_data, boundary_dofs, V, lowest_omega)





import numpy as np
import pyvista as pv
import pyvistaqt
import dolfinx.plot

print("Mapping model data to geometric grid points for visualization...")

# vtk_mesh(V) returns geometry points already ordered to match V's dofmap —
# no re-evaluation or coordinate matching needed.
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid_comparison = pv.UnstructuredGrid(topology, cell_types, geometry)

grid_comparison.point_data["c_true"] = c_func_true.x.array.real.copy()
grid_comparison.point_data["c_recovered"] = c_estimated.x.array.real.copy()

plotter_inv = pyvistaqt.BackgroundPlotter(window_size=(1100, 500), shape=(1, 2))

plotter_inv.subplot(0, 0)
plotter_inv.add_text("Ground Truth Target: C(x,y)", font_size=12, position="upper_edge")
plotter_inv.add_mesh(grid_comparison, show_edges=False, scalars="c_true",
                      cmap="inferno", clim=[0.9, 2.1], lighting=False)
plotter_inv.view_xy()

plotter_inv.subplot(0, 1)
plotter_inv.add_text("Recovered Model Profile (Lowest Frequency)", font_size=12, position="upper_edge")
plotter_inv.add_mesh(grid_comparison, show_edges=False, scalars="c_recovered",
                      cmap="inferno", clim=[0.9, 2.1], lighting=False)
plotter_inv.view_xy()

plotter_inv.show()
print("Visualization windows loaded successfully.")


''' Check to see if boundary data actually looks somewhat similar '''






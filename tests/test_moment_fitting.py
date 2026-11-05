import numpy as np
import pytest

from immersed_elasticity.moment_fitting import compress_rule
from immersed_elasticity.quadrature import box_rule, cut_rule
from immersed_elasticity.solver import Material, structured_mesh, assemble, element_kinematics


def test_cut_cell_compression_preserves_polynomial_integrals():
    bounds = (0, 1, 0, 1)
    points, weights = cut_rule(bounds, .2, depth=6, alpha=0)
    fitted_points, fitted_weights = compress_rule(points, weights, bounds)
    assert np.all(fitted_weights > 0)
    assert len(fitted_weights) < np.count_nonzero(weights)
    assert np.all(np.sum(fitted_points**2, axis=1) >= .2**2)
    for i in range(3):
        for j in range(3):
            reference = weights @ (points[:, 0]**i * points[:, 1]**j)
            actual = fitted_weights @ (fitted_points[:, 0]**i * fitted_points[:, 1]**j)
            np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)


def test_translated_rectangle_matches_exact_polynomial_integrals():
    bounds = (2, 4, -3, -1)
    points, weights = box_rule(bounds, order=5)
    fitted_points, fitted_weights = compress_rule(points, weights, bounds)
    for i in range(3):
        for j in range(3):
            exact = ((4**(i+1) - 2**(i+1)) / (i+1)
                     * ((-1)**(j+1) - (-3)**(j+1)) / (j+1))
            actual = fitted_weights @ (fitted_points[:, 0]**i * fitted_points[:, 1]**j)
            np.testing.assert_allclose(actual, exact, rtol=1e-10, atol=1e-12)


def test_compressed_rule_preserves_assembled_q4_stiffness():
    # Include the fictitious stiffness used by the full solver, and a non-square
    # cell, to check that the moment space covers its actual stiffness integrand.
    bounds = (0, 2, 0, 1)
    mesh = structured_mesh(1, 1, length=2, height=1, radius=.3)
    material = Material(thickness=2)
    reference, _ = assemble(mesh, material, depth=6, alpha=1e-6)
    points, weights = cut_rule(bounds, .3, depth=6, alpha=1e-6)
    fitted_points, fitted_weights = compress_rule(points, weights, bounds, degree=2)
    _, matrices = element_kinematics(mesh.nodes[mesh.cells[0]], fitted_points)
    fitted_stiffness = np.zeros((8, 8))
    for matrix, weight in zip(matrices, fitted_weights):
        fitted_stiffness += matrix.T @ material.D @ matrix * weight * material.thickness
    # Account for the mesh's global DOF order, which differs from element order.
    from immersed_elasticity.solver import dofs
    element_dofs = dofs(mesh.cells[0])
    reference_stiffness = reference.toarray()[np.ix_(element_dofs, element_dofs)]
    relative_error = (np.linalg.norm(fitted_stiffness - reference_stiffness)
                      / np.linalg.norm(reference_stiffness))
    assert relative_error < 1e-10


def test_full_plate_compression_preserves_displacements_and_reactions():
    from immersed_elasticity.reference import kirsch_traction
    from immersed_elasticity.solver import boundary_traction, solve, symmetry_constraints

    mesh = structured_mesh(8, 8, radius=1)
    material = Material()
    force = boundary_traction(mesh, material, kirsch_traction)
    constraints = symmetry_constraints(mesh)
    reference, original_count = assemble(mesh, material, depth=4)
    fitted, compressed_count = assemble(mesh, material, depth=4, method="moment-fitting")
    assert compressed_count < original_count
    reference_u, reference_r = solve(reference, force, constraints)
    fitted_u, fitted_r = solve(fitted, force, constraints)
    assert np.linalg.norm(fitted_u-reference_u) / np.linalg.norm(reference_u) < 1e-10
    fixed = list(constraints)
    assert np.linalg.norm(fitted_r[fixed]-reference_r[fixed]) / np.linalg.norm(reference_r[fixed]) < 1e-10


def test_failed_fit_falls_back_to_quadtree(monkeypatch):
    import immersed_elasticity.solver as solver

    def failed_fit(*args, **kwargs):
        raise RuntimeError("Fit failed")

    monkeypatch.setattr(solver, "compress_rule", failed_fit)
    mesh = structured_mesh(2, 2, radius=1)
    material = Material()
    reference, original_count = assemble(mesh, material)
    with pytest.warns(RuntimeWarning, match="used quadtree"):
        fitted, count = assemble(mesh, material, method="moment-fitting")
    np.testing.assert_array_equal(fitted.toarray(), reference.toarray())
    assert count == original_count

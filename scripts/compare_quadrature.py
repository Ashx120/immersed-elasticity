"""Compare both integration methods on an identical Kirsch plate model."""

from pathlib import Path
from time import perf_counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from immersed_elasticity.reference import kirsch_traction
from immersed_elasticity.solver import (
    Material, assemble, boundary_traction, solve, structured_mesh, symmetry_constraints,
)


def main():
    mesh = structured_mesh(16, 16, radius=1.0)
    material = Material()
    force = boundary_traction(mesh, material, kirsch_traction)
    constraints = symmetry_constraints(mesh)
    results = {}
    for method in ("quadtree", "moment-fitting"):
        start = perf_counter()
        stiffness, count = assemble(mesh, material, depth=4, method=method)
        seconds = perf_counter() - start
        displacement, residual = solve(stiffness, force, constraints)
        results[method] = stiffness, displacement, residual
        print(f"{method}: {count} integration points, assembly including setup = {seconds:.4f} s")

    reference_K, reference_u, reference_r = results["quadtree"]
    fitted_K, fitted_u, fitted_r = results["moment-fitting"]
    matrix_error = (np.linalg.norm((fitted_K - reference_K).data)
                    / np.linalg.norm(reference_K.data))
    displacement_error = np.linalg.norm(fitted_u - reference_u) / np.linalg.norm(reference_u)
    fixed = list(constraints)
    reaction_error = (np.linalg.norm(fitted_r[fixed] - reference_r[fixed])
                      / np.linalg.norm(reference_r[fixed]))
    print(f"Relative stiffness difference:    {matrix_error:.3e}")
    print(f"Relative displacement difference: {displacement_error:.3e}")
    print(f"Relative reaction difference:     {reaction_error:.3e}")
    print("Single-run timings include quadrature generation and fitting; they are diagnostic.")


if __name__ == "__main__":
    main()

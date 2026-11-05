"""First v2 experiment: preserve cut-cell area and moments with fewer points."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from immersed_elasticity.moment_fitting import compress_rule
from immersed_elasticity.quadrature import cut_rule
from immersed_elasticity.solver import Material, element_kinematics


def element_stiffness(nodes, material, points, weights):
    """Integrate the physical part of this rectangular Q4 element."""
    _, strain_matrices = element_kinematics(nodes, points)
    stiffness = np.zeros((8, 8))
    for strain_matrix, weight in zip(strain_matrices, weights):
        stiffness += strain_matrix.T @ material.D @ strain_matrix * weight
    return stiffness * material.thickness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomial degree in each coordinate; 0 fits area only.")
    args = parser.parse_args()
    bounds = (0, 1, 0, 1)
    radius = 0.2
    points, weights = cut_rule(bounds, radius, depth=6, alpha=0)
    fitted_points, fitted_weights = compress_rule(points, weights, bounds, degree=args.degree)

    exact_area = 1 - np.pi * radius**2 / 4
    print(f"Quadtree points generated: {len(points)}")
    print(f"Quadtree points with positive weight: {np.count_nonzero(weights)}")
    print(f"Fitted points retained: {len(fitted_points)}")
    print(f"Smallest fitted weight: {fitted_weights.min():.6e}")
    print(f"Exact material area: {exact_area:.9f} mm^2")
    print(f"Quadtree area:       {weights.sum():.9f} mm^2")
    print(f"Fitted area:         {fitted_weights.sum():.9f} mm^2")
    # x^2 is fitted for degree >= 2; exp(x+y) probes beyond the polynomial space.
    for label, function in [("x^2", lambda p: p[:, 0]**2),
                             ("exp(x+y)", lambda p: np.exp(p.sum(axis=1)))]:
        source = weights @ function(points)
        compressed = fitted_weights @ function(fitted_points)
        print(f"Integral of {label}: quadtree={source:.9f}, fitted={compressed:.9f}")

    nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    material = Material()
    reference_stiffness = element_stiffness(nodes, material, points, weights)
    fitted_stiffness = element_stiffness(nodes, material, fitted_points, fitted_weights)
    relative_difference = (np.linalg.norm(fitted_stiffness - reference_stiffness)
                           / np.linalg.norm(reference_stiffness))
    print(f"Relative stiffness difference (Frobenius norm): {relative_difference:.3e}")
    print("Both matrices integrate material outside the hole (alpha=0).")


if __name__ == "__main__":
    main()

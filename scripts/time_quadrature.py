"""Time cached quadrature reuse for element stiffness integration.

This controlled experiment excludes sparse global assembly, the linear solve,
and constitutive history updates. Both rules recompute Q4 kinematics each pass.
"""

import argparse
import math
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from immersed_elasticity.moment_fitting import compress_rule
from immersed_elasticity.quadrature import classify, cut_rule
from immersed_elasticity.solver import Material, element_kinematics, structured_mesh


def integrate_all(nodes, rules, constitutive_matrix, thickness):
    """Return one 8x8 stiffness matrix per element, without global assembly."""
    matrices = []
    for coordinates, (points, weights) in zip(nodes, rules):
        _, strain_matrices = element_kinematics(coordinates, points)
        stiffness = np.einsum("qia,ij,qjb,q->ab", strain_matrices,
                              constitutive_matrix, strain_matrices, weights)
        matrices.append(stiffness * thickness)
    return np.asarray(matrices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.trials < 1 or args.repeats < 1:
        parser.error("trials and repeats must be positive integers")

    mesh = structured_mesh(16, 16, radius=1)
    material = Material()
    constitutive_matrix = material.D
    nodes = [mesh.nodes[cell] for cell in mesh.cells]
    bounds = [(xy[0, 0], xy[1, 0], xy[0, 1], xy[3, 1]) for xy in nodes]
    setup_times = {"quadtree": [], "moment-fitting": []}
    evaluation_times = {"quadtree": [], "moment-fitting": []}

    for trial in range(args.trials):
        start = perf_counter()
        reference_rules = [cut_rule(box, mesh.radius, depth=4, alpha=1e-6) for box in bounds]
        reference_seconds = perf_counter() - start
        start = perf_counter()
        fitted_rules = []
        for box, (points, weights) in zip(bounds, reference_rules):
            if classify(box, mesh.radius) == 0:
                points, weights = compress_rule(points, weights, box, degree=2)
            fitted_rules.append((points, weights))
        fitting_seconds = perf_counter() - start
        setup_times["quadtree"].append(reference_seconds)
        setup_times["moment-fitting"].append(reference_seconds + fitting_seconds)

        rules_by_method = {"quadtree": reference_rules, "moment-fitting": fitted_rules}
        # Untimed verification also warms up both paths before measurement.
        reference = integrate_all(nodes, reference_rules, constitutive_matrix, material.thickness)
        fitted = integrate_all(nodes, fitted_rules, constitutive_matrix, material.thickness)
        relative_difference = np.linalg.norm(fitted - reference) / np.linalg.norm(reference)
        if relative_difference > 1e-10:
            raise RuntimeError(f"Stiffness comparison failed: {relative_difference:.3e}")

        methods = list(rules_by_method)
        if trial % 2:
            methods.reverse()
        for method in methods:
            start = perf_counter()
            for _ in range(args.repeats):
                integrate_all(nodes, rules_by_method[method], constitutive_matrix, material.thickness)
            evaluation_times[method].append((perf_counter() - start) / args.repeats)

    print(f"16x16 plate, depth 4; medians of {args.trials} trials, {args.repeats} passes per trial")
    for method, rules in rules_by_method.items():
        count = sum(len(weights) for _, weights in rules)
        print(f"{method}: {count} points, setup={np.median(setup_times[method]):.6f} s, "
              f"one integration pass={np.median(evaluation_times[method]):.6f} s")

    extra_setup = np.median(setup_times["moment-fitting"]) - np.median(setup_times["quadtree"])
    saving = np.median(evaluation_times["quadtree"]) - np.median(evaluation_times["moment-fitting"])
    if saving > 0:
        passes = max(1, math.floor(extra_setup / saving) + 1)
        print(f"Estimated first cheaper pass count: {passes}")
    else:
        print("No measured break-even: fitted-rule integration was not faster.")
    print(f"Relative element-stiffness difference: {relative_difference:.3e}")
    print("Times cover rule setup and element integration only; timings vary by run.")


if __name__ == "__main__":
    main()

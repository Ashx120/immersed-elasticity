"""Compress an existing positive quadrature rule by fitting polynomial moments."""

import numpy as np
from scipy.optimize import nnls


def compress_rule(points, weights, bounds, degree=2, tolerance=1e-10):
    """Preserve tensor-product moments through ``degree`` in each coordinate.

    Candidates are the positive-weight points of the input rule. Thus an exact
    feasible solution exists before compression. Coordinates are scaled to the
    parent cell's [-1, 1]^2 domain for conditioning. The fitted rule inherits
    geometry-integration error from its input rule.
    """
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or weights.shape != (len(points),):
        raise ValueError("Require points with shape (n, 2) and weights with shape (n,).")
    if (not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError("Require finite points and non-negative weights with positive total mass.")
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("Require an integer degree >= 0.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Require a positive finite tolerance.")
    x0, x1, y0, y1 = bounds
    if not np.all(np.isfinite(bounds)) or x1 <= x0 or y1 <= y0:
        raise ValueError("Require finite bounds with positive width and height.")

    positive = weights > 0
    candidates = points[positive]
    source_weights = weights[positive]
    xi = 2 * (candidates[:, 0] - x0) / (x1 - x0) - 1
    eta = 2 * (candidates[:, 1] - y0) / (y1 - y0) - 1
    moment_matrix = np.array([
        xi**i * eta**j
        for i in range(degree + 1)
        for j in range(degree + 1)
    ])
    mass = source_weights.sum()
    target = moment_matrix @ (source_weights / mass)
    fitted, _ = nnls(moment_matrix, target, maxiter=10 * len(candidates))
    relative_residual = np.linalg.norm(moment_matrix @ fitted - target) / np.linalg.norm(target)
    if relative_residual > tolerance:
        raise RuntimeError(f"Moment fit residual {relative_residual:.3e} exceeds tolerance.")
    retained = fitted > 0
    return candidates[retained], fitted[retained] * mass

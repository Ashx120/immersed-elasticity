"""Positive Gaussian quadrature on recursively subdivided cut rectangles."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def gauss(order=2):
    return np.polynomial.legendre.leggauss(order)


def box_rule(bounds, order=2):
    """Return physical coordinates and area weights for an axis-aligned box."""
    x0, x1, y0, y1 = bounds
    g, w = gauss(order)
    xx, yy = np.meshgrid((x0+x1)/2 + (x1-x0)*g/2,
                         (y0+y1)/2 + (y1-y0)*g/2, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel())), (np.outer(w, w)*(x1-x0)*(y1-y0)/4).ravel()


def classify(bounds, radius):
    """Return +1 for physical, -1 for fictitious, 0 for intersected boxes.

    The circular hole is centered at the origin; works beyond the first quadrant.
    Distance bounds avoid missing a curved interface between sampled corners.
    """
    if radius <= 0:
        return 1
    x0, x1, y0, y1 = bounds
    near_x = max(x0, 0, -x1)
    near_y = max(y0, 0, -y1)
    far_x, far_y = max(abs(x0), abs(x1)), max(abs(y0), abs(y1))
    if near_x**2 + near_y**2 >= radius**2:
        return 1
    if far_x**2 + far_y**2 <= radius**2:
        return -1
    return 0


def cut_rule(bounds, radius, depth=3, alpha=1e-6, order=2):
    """Integrate physical + alpha-scaled fictitious regions of one active cell.

    Only interface boxes recurse. At terminal cut boxes the indicator is
    sampled at Gaussian points, so geometry integration remains approximate.
    """
    chunks = []

    def visit(box, level):
        kind = classify(box, radius)
        if kind or level == depth:
            points, weights = box_rule(box, order)
            physical = np.sum(points**2, axis=1) >= radius**2
            chunks.append((points, weights*np.where(physical, 1.0, alpha)))
            return
        x0, x1, y0, y1 = box
        xm, ym = (x0+x1)/2, (y0+y1)/2
        for sub in [(x0,xm,y0,ym), (xm,x1,y0,ym),
                    (x0,xm,ym,y1), (xm,x1,ym,y1)]:
            visit(sub, level+1)

    visit(bounds, 0)
    return np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])

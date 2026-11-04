"""Kirsch's infinite-plate stress field, restricted to a finite square.

Apply its exact tractions on the square outer edges. This makes Kirsch a
valid stress reference on the truncated domain without a finite-width error.
Remote tension acts in the x direction; no displacement formula is needed.
"""

import numpy as np


def kirsch_stress(points, radius=1.0, tension=100.0):
    x, y = np.asarray(points).T
    r2 = x*x+y*y
    if radius <= 0 or np.any(r2 < radius**2*(1-1e-12)):
        raise ValueError("Kirsch stress requires radius > 0 and points outside the hole.")
    theta = np.arctan2(y,x)
    q = radius**2/r2
    c, s = np.cos(theta), np.sin(theta)
    rr = tension/2*((1-q)+(1-4*q+3*q*q)*np.cos(2*theta))
    tt = tension/2*((1+q)-(1+3*q*q)*np.cos(2*theta))
    rt = -tension/2*(1+2*q-3*q*q)*np.sin(2*theta)
    return np.column_stack((rr*c*c+tt*s*s-2*rt*s*c,
                            rr*s*s+tt*c*c+2*rt*s*c,
                            (rr-tt)*s*c+rt*(c*c-s*s)))


def kirsch_traction(points, normal, radius=1.0, tension=100.0):
    stress = kirsch_stress(points,radius,tension)
    return np.column_stack((stress[:,0]*normal[0]+stress[:,2]*normal[1],
                            stress[:,2]*normal[0]+stress[:,1]*normal[1]))

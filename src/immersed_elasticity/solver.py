"""Q4 elements, engineering shear strain, sparse assembly, and exact constraints.

Units are user-consistent; the example uses mm, N, MPa (= N/mm^2).
Only axis-aligned Cartesian elements and small-strain plane stress are supported.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .quadrature import classify, cut_rule, gauss


@dataclass(frozen=True)
class Material:
    young: float = 210000.0
    poisson: float = 0.3
    thickness: float = 1.0

    def __post_init__(self):
        if not (np.isfinite(self.young) and self.young > 0 and
                np.isfinite(self.thickness) and self.thickness > 0 and
                np.isfinite(self.poisson) and -1 < self.poisson < 0.5):
            raise ValueError("Require E > 0, thickness > 0, and -1 < nu < 0.5.")

    @property
    def D(self):
        v = self.poisson
        return self.young/(1-v*v)*np.array([[1,v,0], [v,1,0], [0,0,(1-v)/2]])


@dataclass
class Mesh:
    nodes: np.ndarray
    cells: np.ndarray
    length: float
    height: float
    radius: float


def structured_mesh(nx, ny, length=4.0, height=4.0, radius=0.0):
    if (nx < 1 or ny < 1 or int(nx) != nx or int(ny) != ny or
            not np.all(np.isfinite([length, height, radius])) or
            length <= 0 or height <= 0 or not 0 <= radius < min(length, height)):
        raise ValueError("Require positive integer mesh counts and 0 <= radius < min(L,H).")
    xx, yy = np.meshgrid(np.linspace(0,length,nx+1), np.linspace(0,height,ny+1))
    nodes = np.column_stack((xx.ravel(), yy.ravel()))
    cells = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i
            c = [a,a+1,a+nx+2,a+nx+1]  # counterclockwise
            xy = nodes[c]
            if classify((xy[0,0],xy[1,0],xy[0,1],xy[3,1]), radius) != -1:
                cells.append(c)
    # Fully fictitious cells and nodes unused by active cells are excluded.
    used, inverse = np.unique(np.asarray(cells).ravel(), return_inverse=True)
    return Mesh(nodes[used], inverse.reshape(-1,4), length, height, radius)


def element_kinematics(xy, points):
    """Return N[q,4] and B[q,3,8] for axis-aligned Q4 elements."""
    hx, hy = xy[1,0]-xy[0,0], xy[3,1]-xy[0,1]
    xi = 2*(points[:,0]-xy[0,0])/hx-1
    eta = 2*(points[:,1]-xy[0,1])/hy-1
    sx, sy = np.array([-1,1,1,-1]), np.array([-1,-1,1,1])
    N = (1+xi[:,None]*sx)*(1+eta[:,None]*sy)/4
    dx = sx*(1+eta[:,None]*sy)/(2*hx)
    dy = sy*(1+xi[:,None]*sx)/(2*hy)
    B = np.zeros((len(points),3,8))
    B[:,0,0::2], B[:,1,1::2] = dx, dy
    B[:,2,0::2], B[:,2,1::2] = dy, dx
    return N, B


def dofs(cell):
    return (2*np.asarray(cell)[:,None]+np.array([0,1])).ravel()


def assemble(mesh, material, depth=3, alpha=1e-6):
    if not (0 < alpha <= 1) or depth < 0 or int(depth) != depth:
        raise ValueError("Require 0 < alpha <= 1 and integer depth >= 0.")
    rows, cols, values = [], [], []
    point_count = 0
    for cell in mesh.cells:
        xy = mesh.nodes[cell]
        bounds = (xy[0,0],xy[1,0],xy[0,1],xy[3,1])
        points, weights = cut_rule(bounds, mesh.radius, depth, alpha)
        _, B = element_kinematics(xy, points)
        ke = np.einsum("qia,ij,qjb,q->ab", B, material.D, B, weights)*material.thickness
        ids = dofs(cell)
        rows.extend(np.repeat(ids,8)); cols.extend(np.tile(ids,8)); values.extend(ke.ravel())
        point_count += len(points)
    n = 2*len(mesh.nodes)
    K = coo_matrix((values,(rows,cols)),shape=(n,n)).tocsr()
    return K, point_count


def boundary_traction(mesh, material, traction, sides=("right","top"), order=4):
    """Consistent edge loads. traction(points, outward_normal) -> [q,2]."""
    force = np.zeros(2*len(mesh.nodes))
    g, w = gauss(order)
    edge_defs = {"right": (1,2,np.array([1.,0.])),
                 "top": (2,3,np.array([0.,1.])),
                 "left": (3,0,np.array([-1.,0.])),
                 "bottom": (0,1,np.array([0.,-1.]))}
    for cell in mesh.cells:
        xy = mesh.nodes[cell]
        for side in sides:
            a,b,normal = edge_defs[side]
            axis = 0 if side in ("left","right") else 1
            limit = {"left":0.,"right":mesh.length,"bottom":0.,"top":mesh.height}[side]
            if not np.all(np.isclose(xy[[a,b],axis],limit,rtol=0,atol=1e-12)):
                continue
            p = (1-g[:,None])*xy[a]/2 + (1+g[:,None])*xy[b]/2
            # Exclude portions inside the hole (outer benchmark edges never cut).
            wt = w*np.linalg.norm(xy[b]-xy[a])/2*material.thickness
            wt *= np.sum(p*p,axis=1) >= mesh.radius**2
            N,_ = element_kinematics(xy,p)
            fe = np.einsum("qi,qj,q->ij", N, traction(p,normal), wt)
            np.add.at(force,dofs(cell),fe.ravel())
    return force


def symmetry_constraints(mesh):
    return {**{2*i:0. for i in np.flatnonzero(np.isclose(mesh.nodes[:,0],0))},
            **{2*i+1:0. for i in np.flatnonzero(np.isclose(mesh.nodes[:,1],0))}}


def solve(K, force, constraints):
    """Eliminate prescribed DOFs; reactions use the original unmodified system."""
    fixed = np.array(sorted(constraints),dtype=int)
    free = np.setdiff1d(np.arange(K.shape[0]),fixed)
    u = np.zeros(K.shape[0])
    u[fixed] = [constraints[i] for i in fixed]
    with warnings.catch_warnings():
        warnings.simplefilter("error",MatrixRankWarning)
        u[free] = spsolve(K[free][:,free],force[free]-K[free][:,fixed]@u[fixed])
    if not np.all(np.isfinite(u)):
        raise RuntimeError("Non-finite displacement; check constraints and conditioning.")
    residual = K@u-force
    return u, residual

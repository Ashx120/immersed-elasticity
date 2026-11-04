import numpy as np
import pytest

from immersed_elasticity.solver import (Material, structured_mesh, assemble, solve,
    dofs, element_kinematics, boundary_traction, symmetry_constraints)
from immersed_elasticity.quadrature import box_rule, cut_rule
from immersed_elasticity.reference import kirsch_stress, kirsch_traction


def test_affine_patch_with_nonzero_dirichlet_data():
    mesh = structured_mesh(3,3,length=2,height=1)
    material = Material()
    K,_ = assemble(mesh,material)
    x,y = mesh.nodes.T
    expected = np.column_stack((.002*x+.003*y+.01, -.001*x+.004*y-.02)).ravel()
    boundary = np.flatnonzero(np.isclose(x,0)|np.isclose(x,2)|np.isclose(y,0)|np.isclose(y,1))
    bc = {j:expected[j] for i in boundary for j in [2*i,2*i+1]}
    u,r = solve(K,np.zeros_like(expected),bc)
    np.testing.assert_allclose(u,expected,atol=1e-14)
    for cell in mesh.cells:
        _,B = element_kinematics(mesh.nodes[cell],mesh.nodes[cell].mean(axis=0)[None,:])
        np.testing.assert_allclose(B[0]@u[dofs(cell)],[.002,.004,.002],atol=1e-14)
    np.testing.assert_allclose(r.reshape(-1,2).sum(axis=0),0,atol=1e-10)
    assert np.max(np.abs((K-K.T).data)) < 1e-9


def test_rectangle_uniaxial_tension_and_poisson_contraction():
    mesh = structured_mesh(4,3,length=3,height=2)
    mat = Material(thickness=2)
    K,_ = assemble(mesh,mat)
    f = boundary_traction(mesh,mat,lambda p,n: np.tile([100.,0.],(len(p),1)),sides=("right",))
    u,r = solve(K,f,symmetry_constraints(mesh))
    exact = np.column_stack((100/mat.young*mesh.nodes[:,0],
                            -mat.poisson*100/mat.young*mesh.nodes[:,1])).ravel()
    np.testing.assert_allclose(u,exact,atol=1e-14)
    np.testing.assert_allclose(f.reshape(-1,2).sum(axis=0),[400,0],atol=1e-10)
    np.testing.assert_allclose((r+f).reshape(-1,2).sum(axis=0),0,atol=1e-10)


def test_cut_quadrature_area_and_positive_weights():
    radius,alpha = .9,1e-6
    exact = 1-(1-alpha)*np.pi*radius**2/4
    errors=[]
    for depth in (2,6):
        _,w = cut_rule((0,1,0,1),radius,depth,alpha)
        assert np.all(w>0)
        errors.append(abs(w.sum()-exact))
    assert errors[1] < errors[0]
    assert errors[1] < 2e-4


def test_kirsch_hole_is_traction_free_and_has_expected_peak():
    angles = np.linspace(0,np.pi/2,21)
    points = np.column_stack((np.cos(angles),np.sin(angles)))
    stress = kirsch_stress(points)
    traction = np.column_stack((stress[:,0]*points[:,0]+stress[:,2]*points[:,1],
                                stress[:,2]*points[:,0]+stress[:,1]*points[:,1]))
    np.testing.assert_allclose(traction,0,atol=1e-12)
    np.testing.assert_allclose(kirsch_stress([[0,1]])[0,0],300)


def test_immersed_kirsch_stress_converges():
    mat=Material()
    errors=[]
    for n in (8,16):
        mesh=structured_mesh(n,n,radius=1)
        K,_=assemble(mesh,mat,depth=4)
        f=boundary_traction(mesh,mat,kirsch_traction)
        u,r=solve(K,f,symmetry_constraints(mesh))
        numerator=denominator=0.
        for cell in mesh.cells:
            xy=mesh.nodes[cell]
            p,w=box_rule((xy[0,0],xy[1,0],xy[0,1],xy[3,1]),order=6)
            keep=np.sum(p*p,axis=1)>=1
            if not np.any(keep):
                continue
            p,w=p[keep],w[keep]
            _,B=element_kinematics(xy,p)
            stress=np.einsum("ij,qjk,k->qi",mat.D,B,u[dofs(cell)])
            ref=kirsch_stress(p)
            # Tensor Frobenius norm: shear appears twice.
            numerator+=np.sum(w[:,None]*(stress-ref)**2*[1,1,2])
            denominator+=np.sum(w[:,None]*ref**2*[1,1,2])
        errors.append(np.sqrt(numerator/denominator))
        np.testing.assert_allclose((r+f).reshape(-1,2).sum(axis=0),0,atol=1e-7)
    assert errors[1]<errors[0]
    assert errors[1]<.15


@pytest.mark.parametrize("kwargs",[{"young":0},{"poisson":.5},{"thickness":-1}])
def test_invalid_material(kwargs):
    with pytest.raises(ValueError):
        Material(**kwargs)

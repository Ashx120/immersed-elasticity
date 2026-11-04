"""Reproducible verification experiment and publication assets for v1."""

import argparse
import csv
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.ticker import ScalarFormatter, NullFormatter
import numpy as np
import scipy

from .quadrature import box_rule, cut_rule
from .reference import kirsch_stress, kirsch_traction
from .solver import (Material, structured_mesh, assemble, boundary_traction,
                     solve, symmetry_constraints, dofs, element_kinematics)


def evaluate(mesh, material, u, evaluation_depth=6):
    """Stress L2 error with a separate, fixed, deeper physical quadrature.

    This evaluates the full stress tensor Frobenius norm, not an energy norm.
    Geometry integration in this error estimate is itself approximate.
    """
    numerator=denominator=area=0.
    for cell in mesh.cells:
        xy=mesh.nodes[cell]
        p,w=cut_rule((xy[0,0],xy[1,0],xy[0,1],xy[3,1]),mesh.radius,
                     depth=evaluation_depth,alpha=0,order=3)
        keep=w>0
        p,w=p[keep],w[keep]
        if not len(w):
            continue
        _,B=element_kinematics(xy,p)
        stress=np.einsum("ij,qjk,k->qi",material.D,B,u[dofs(cell)])
        ref=kirsch_stress(p,mesh.radius)
        numerator+=np.sum(w[:,None]*(stress-ref)**2*[1,1,2])
        denominator+=np.sum(w[:,None]*ref**2*[1,1,2])
        area+=w.sum()
    return float(np.sqrt(numerator/denominator)),float(area)


def run_case(n, depth=4, alpha=1e-6):
    mat=Material()
    mesh=structured_mesh(n,n,radius=1.)
    start=perf_counter()
    K,nq=assemble(mesh,mat,depth,alpha)
    assembly_seconds=perf_counter()-start
    f=boundary_traction(mesh,mat,kirsch_traction)
    bc=symmetry_constraints(mesh)
    start=perf_counter()
    u,reactions=solve(K,f,bc)
    solve_seconds=perf_counter()-start
    error,area=evaluate(mesh,mat,u)
    free=np.setdiff1d(np.arange(len(u)),list(bc))
    force_scale=np.linalg.norm(f)
    result={"mesh_n":int(n),"cells":len(mesh.cells),"dofs":len(u),
            "quadrature_depth":int(depth),"alpha":float(alpha),
            "quadrature_points":nq,"relative_stress_L2_error":error,
            "integrated_physical_area":area,"exact_physical_area":16-np.pi/4,
            "relative_free_residual":float(np.linalg.norm(reactions[free])/force_scale),
            "relative_force_imbalance":float(np.linalg.norm((reactions+f).reshape(-1,2).sum(axis=0))/force_scale),
            "strain_energy_N_mm":float(.5*u@(K@u)),
            "relative_energy_identity_error":float(abs(u@(K@u)-u@f)/abs(u@f)),
            "assembly_seconds":assembly_seconds,"solve_seconds":solve_seconds}
    return result,(mesh,mat,u)


def plot_fields(state,path,*,compare_kirsch=True):
    mesh,mat,u=state
    locations=[];displacements=[];stresses=[]
    for cell in mesh.cells:
        xy=mesh.nodes[cell]
        p,_=box_rule((xy[0,0],xy[1,0],xy[0,1],xy[3,1]),order=5)
        p=p[np.sum(p*p,axis=1)>=mesh.radius**2]
        if not len(p):
            continue
        N,B=element_kinematics(xy,p)
        locations.append(p)
        displacements.append(N@u[dofs(cell)].reshape(4,2))
        stresses.append(np.einsum("ij,qjk,k->qi",mat.D,B,u[dofs(cell)]))
    p=np.concatenate(locations)
    disp=np.concatenate(displacements)
    stress=np.concatenate(stresses)
    fields=[(np.linalg.norm(disp,axis=1)*1000,"Displacement magnitude","micrometres","viridis"),
            (stress[:,0],"Computed stress: sigma_xx","MPa","viridis")]
    title="Custom model | Q4 plane stress"
    if compare_kirsch:
        ref=kirsch_stress(p,mesh.radius)
        fields.append((stress[:,0]-ref[:,0],"Stress error: sigma_xx","MPa","coolwarm"))
        title="Quarter plate with a hole | Q4 plane stress | exact Kirsch outer tractions"
    fig,axs=plt.subplots(1,len(fields),figsize=(4.7*len(fields),4.6),layout="constrained")
    for ax,(values,title,label,cmap) in zip(axs,fields):
        kwargs={}
        if cmap=="coolwarm":
            lim=max(abs(values.min()),abs(values.max()))
            kwargs={"vmin":-lim,"vmax":lim}
        sc=ax.scatter(*p.T,c=values,s=6,marker="s",cmap=cmap,rasterized=True,**kwargs)
        ax.add_patch(Circle((0,0),mesh.radius,facecolor="white",edgecolor="#334155",lw=1.1))
        ax.set(xlim=(0,mesh.length),ylim=(0,mesh.height),aspect="equal",xlabel="x [mm]",ylabel="y [mm]",title=title)
        fig.colorbar(sc,ax=ax,shrink=.75,label=label)
    fig.suptitle(title,fontsize=13)
    fig.savefig(path,dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(6,6),layout="constrained")
    segments=[mesh.nodes[cell[[0,1,2,3,0]]] for cell in mesh.cells]
    ax.add_collection(LineCollection(segments,colors="#94a3b8",linewidths=.5))
    cut=[cell for cell in mesh.cells if np.any(np.sum(mesh.nodes[cell]**2,axis=1)<mesh.radius**2)]
    for cell in cut:
        xy=mesh.nodes[cell]
        ax.fill(xy[:,0],xy[:,1],color="#fbbf24",alpha=.5)
    ax.add_patch(Circle((0,0),mesh.radius,facecolor="white",edgecolor="#0f172a",lw=1.5))
    ax.set(xlim=(0,mesh.length),ylim=(0,mesh.height),aspect="equal",xlabel="x [mm]",ylabel="y [mm]",
           title="Cartesian background mesh" + ("\nAmber: cells cut by the circular hole" if cut else ""))
    fig.savefig(path.with_name("mesh.png"),dpi=180)
    plt.close(fig)


def plot_convergence(results,path):
    fig,ax=plt.subplots(figsize=(6,4.5),layout="constrained")
    ax.loglog([r["dofs"] for r in results],
              [100*r["relative_stress_L2_error"] for r in results],"o-",color="#2563eb",lw=2)
    ax.set(xlabel="Active displacement DOFs",ylabel="Relative stress L2 error [%]",
           title="Convergence against the exact Kirsch stress field")
    ax.set_xticks([r["dofs"] for r in results])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([2,3,4,6,8])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True,which="both",alpha=.25)
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh",nargs="+",type=int,default=[8,16,32])
    parser.add_argument("--depth",type=int,default=4)
    parser.add_argument("--alpha",type=float,default=1e-6)
    parser.add_argument("--sensitivity",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("results"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for n in sorted(set(args.mesh)):
        record,state=run_case(n,args.depth,args.alpha)
        results.append(record)
        print(f"n={n:3d}: {record['dofs']:5d} DOFs, stress error={100*record['relative_stress_L2_error']:.3f}%, "
              f"free residual={record['relative_free_residual']:.2e}",flush=True)
    plot_fields(state,args.output/"fields.png")
    plot_convergence(results,args.output/"convergence.png")
    sensitivity=[]
    if args.sensitivity:
        for alpha in (1e-4,1e-6,1e-8):
            record,_=run_case(16,args.depth,alpha)
            sensitivity.append({"study":"fictitious_stiffness",**record})
        for depth in (2,4,6):
            record,_=run_case(16,depth,args.alpha)
            sensitivity.append({"study":"quadrature_depth",**record})
    payload={"environment":{"python":platform.python_version(),"numpy":np.__version__,
                             "scipy":scipy.__version__,"matplotlib":matplotlib.__version__,
                             "platform":platform.platform()},
             "configuration":{"E_MPa":210000.,"nu":.3,"thickness_mm":1.,
                              "quarter_side_mm":4.,"hole_radius_mm":1.,"remote_tension_MPa":100.,
                              "error_evaluation_depth":6,"error_evaluation_gauss_order":3},
             "convergence":results,"sensitivity":sensitivity}
    (args.output/"metrics.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    with (args.output/"convergence.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)


if __name__=="__main__":
    main()

"""Edit the settings below, then run this file from the project folder."""

from pathlib import Path

import numpy as np

from immersed_elasticity.benchmark import plot_fields
import matplotlib.pyplot as plt

from immersed_elasticity.solver import (
    Material, assemble, boundary_traction, solve, structured_mesh,
)

# Geometry and mesh. Use radius=0 for a solid rectangle. Lengths are in mm.
GEOMETRY = dict(nx=24, ny=16, length=6.0, height=4.0, radius=1)

# Young's modulus is in MPa and thickness is in mm.
MATERIAL = dict(young=70_000, poisson=0.33, thickness=1.0)

# Constant edge tractions (tx, ty) in MPa, in global x and y directions.
# Omitted edges have zero applied traction. The circular boundary is unloaded.
TRACTIONS = {"right": (100.0, 0.0)}

# Prescribed edge displacements (ux, uy) in mm. None leaves a component free.
# For displacement loading, set TRACTIONS = {} and add "right": (0.01, None).
DISPLACEMENTS = {
    "left": (0.0, None),
    "bottom": (None, 0.0),
}

# Cut-cell integration settings.
# Choose "quadtree" or "moment-fitting". Fitting compresses cut cells only.
QUADRATURE = dict(depth=6, alpha=1e-6, method="moment-fitting")

# The field and mesh plots use the base grid. The convergence plot also solves
# finer grids with these multipliers applied to both nx and ny.
REFINEMENT_MULTIPLIERS = (2, 4)
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "playground"


def run_model(geometry, *, report=False):
    mesh = structured_mesh(**geometry)
    material = Material(**MATERIAL)
    x, y = mesh.nodes.T
    edge_nodes = {
        "left": np.flatnonzero(np.isclose(x, 0.0)),
        "right": np.flatnonzero(np.isclose(x, mesh.length)),
        "bottom": np.flatnonzero(np.isclose(y, 0.0)),
        "top": np.flatnonzero(np.isclose(y, mesh.height)),
    }

    constraints = {}
    for edge, components in DISPLACEMENTS.items():
        for axis, value in enumerate(components):
            if value is None:
                continue
            for node in edge_nodes[edge]:
                dof = 2 * node + axis
                if dof in constraints and constraints[dof] != value:
                    raise ValueError(f"Conflicting displacements at node {node} on {edge} edge.")
                constraints[dof] = value

    K, point_count = assemble(mesh, material, **QUADRATURE)
    force = np.zeros(2 * len(mesh.nodes))
    for edge, components in TRACTIONS.items():
        def constant_traction(points, normal):
            return np.tile(components, (len(points), 1))

        force += boundary_traction(mesh, material, constant_traction, sides=(edge,))

    displacement, residual = solve(K, force, constraints)
    energy = float(0.5 * displacement @ (K @ displacement))
    state = mesh, material, displacement
    if not report:
        return state, energy
    nodal_displacement = displacement.reshape(-1, 2)
    reactions = np.zeros_like(residual)
    fixed = list(constraints)
    reactions[fixed] = residual[fixed]
    applied = force.reshape(-1, 2).sum(axis=0)
    support = reactions.reshape(-1, 2).sum(axis=0)

    print(f"Quadrature: {QUADRATURE.get('method', 'quadtree')}")
    print(f"Active elements: {len(mesh.cells)} | Integration points: {point_count}")
    print(f"Mean right-edge ux: {nodal_displacement[edge_nodes['right'], 0].mean():+.6f} mm")
    print(f"Applied force (x, y):       ({applied[0]:+.3f}, {applied[1]:+.3f}) N")
    print(f"Total support force (x, y): ({support[0]:+.3f}, {support[1]:+.3f}) N")
    if any(2 * node in constraints for node in edge_nodes["right"]):
        right_reaction = reactions[2 * edge_nodes["right"]].sum()
        print(f"Right-edge x reaction: {right_reaction:+.3f} N")
    print(f"Force imbalance: {np.linalg.norm(applied + support):.3e} N")
    return state, energy


def main():
    state, energy = run_model(GEOMETRY, report=True)
    dof_counts = [len(state[2])]
    energies = [energy]
    for multiplier in REFINEMENT_MULTIPLIERS:
        geometry = {**GEOMETRY, "nx": GEOMETRY["nx"] * multiplier,
                    "ny": GEOMETRY["ny"] * multiplier}
        refined_state, refined_energy = run_model(geometry)
        dof_counts.append(len(refined_state[2]))
        energies.append(refined_energy)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_fields(state, OUTPUT_DIR / "fields.png", compare_kirsch=False)

    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    ax.semilogx(dof_counts, energies, "o-", color="#2563eb")
    ax.set(xlabel="Active displacement DOFs", ylabel="Strain energy [N mm]",
           title="Custom model mesh refinement")
    ax.set_xticks(dof_counts, labels=[str(count) for count in dof_counts])
    ax.grid(True, alpha=0.25)
    fig.savefig(OUTPUT_DIR / "convergence.png", dpi=180)
    plt.close(fig)
    print(f"Saved fields.png, mesh.png, and convergence.png to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

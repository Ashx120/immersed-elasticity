"""Small-strain plane-stress FEM with Cartesian immersed geometry."""

from .solver import Material, Mesh, assemble, solve, structured_mesh

__all__ = ["Material", "Mesh", "assemble", "solve", "structured_mesh"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jiménez fintie element analysis."""

import sys

import numpy as np

from fea.assembly import rhs_vector, stiffness_matrix
from fea.material import Material
from fea.mesh import Mesh


def main() -> int:
    """Entry point to the code."""

    # construct the mesh
    omega = Mesh.grid_mesh(m=4, n=16, L=1e-2, H=4e-2)

    # set up a material
    steel = Material(E=200e9, nu=0.28)

    # construct the Dirichlet boundary
    bottom = []
    left = []
    top = []
    for i in omega.boundary():
        if np.isclose(a=omega.coordinates[i, 0], b=0.0):
            left.append(2*i)
        if np.isclose(a=omega.coordinates[i, 1], b=0.0):
            bottom.append(2*i + 1)
        if np.isclose(a=omega.coordinates[i, 1], b=4e-2):
            top.append(2*i + 1)

    # assemble the finite element system
    body_force = np.zeros(shape=2, dtype=np.float64)
    K = stiffness_matrix(mesh=omega, material=steel, plane_stress=True)
    f = rhs_vector(mesh=omega, b=body_force)

    # apply Dirichlet boundary conditions
    for dof in bottom + left:
        f -= 0.0*K[dof, :]
        f[dof] = 0.0
        K[dof, :] = K[:, dof] = 0.0
        K[dof, dof] = 1.0
    for dof in top:
        f -= 1e-6*K[dof, :]
        f[dof] = 1e-6
        K[dof, :] = K[:, dof] = 0.0
        K[dof, dof] = 1.0

    # solve the finite element system of equations
    u = np.linalg.solve(a=K, b=f)
    print(u)

    return 0


if __name__ == '__main__':
    sys.exit(main())

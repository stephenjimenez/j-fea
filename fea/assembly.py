#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tools for assembling the finite element system of equations."""

import numpy as np
from numpy.typing import NDArray

from .material import Material
from .mesh import Mesh
import shape


def stiffness_matrix(mesh: Mesh, material: Material,
                     plane_stress: bool = True) -> NDArray[np.float64]:
    """Assemble the global stiffness matrix."""

    # assemble material stiffness matrix
    if plane_stress:
        D = material.plane_stress_matrix()
    else:
        D = material.plane_strain_matrix()

    # loop over all elements to populate global stiffness matrix
    n_dofs = 2*mesh.n_nodes
    K_global = np.zeros(shape=(n_dofs, n_dofs), dtype=np.float64)
    for i in range(mesh.n_elements):

        # construct stiffness matrix for element "i" by performing
        # Gauss quadrature
        K = np.zeros(shape=(8, 8), dtype=np.float64)
        x = mesh.coordinates[mesh.topology[i]]
        for j in range(4):
            xi, eta = shape.Q4_GAUSS_POINTS[j]
            w = shape.Q4_GAUSS_WEIGHTS[j]
            dphi = shape.q4gradient(xi=xi, eta=eta)
            jac = dphi @ x
            det = np.linalg.det(a=jac)
            if det <= 0.0:
                raise ValueError('Determinant of Jacobian is <= 0.0!')
            dphi_dx = np.linalg.solve(a=jac, b=dphi)
            B = np.zeros(shape=(3, 8), dtype=np.float64)
            for k in range(4):
                B[0, 2*k] = B[2, 2*k + 1] = dphi_dx[0, i]
                B[1, 2*k + 1] = B[2, 2*k] = dphi_dx[1, i]
            K += w*(B.T @ D @ B)*det

        # pack the global stiffness matrix
        dofs = np.zeros(shape=8, dtype=np.int32)
        for j in range(4):
            dofs[2*j] = 2*mesh.topology[i, j]
            dofs[2*j + 1] = 2*mesh.topology[i, j] - 1
        for j in range(8):
            for k in range(8):
                K_global[dofs[j], dofs[k]] += K[j, k]

    return K_global

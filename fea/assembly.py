#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tools for assembling the finite element system of equations."""

import numpy as np
from numpy.typing import NDArray

from .material import Material
from .mesh import Mesh
from .shape import Q4_GAUSS_POINTS, Q4_GAUSS_WEIGHTS, q4gradient, q4shape


def rhs_vector(mesh: Mesh, b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Assemble the global right-hand side (RHS) vector."""

    # iterate over all elements to populate global RHS vector
    n_dofs = 2*mesh.n_nodes
    f_global = np.zeros(shape=n_dofs, dtype=np.float64)
    if np.count_nonzero(a=b) > 0:
        for i in range(mesh.n_elements):

            # construct RHS vector for element "i" by performing
            # Gauss quadrature
            f = np.zeros(shape=8, dtype=np.float64)
            x = mesh.coordinates[mesh.topology[i]]
            for j in range(4):
                xi, eta = Q4_GAUSS_POINTS[j]
                w = Q4_GAUSS_WEIGHTS[j]
                dphi = q4gradient(xi=xi, eta=eta)
                phi = q4shape(xi=xi, eta=eta)
                jac = dphi @ x
                det = np.linalg.det(a=jac)
                if det <= 0.0:
                    raise ValueError('Determinant of Jacobian is <= 0')
                f[0:8:2] += w*phi*b[0]*det
                f[1:8:2] += w*phi*b[1]*det

            # pack the global RHS vector
            dofs = np.zeros(shape=8, dtype=np.int32)
            for j in range(4):
                dofs[2*j] = 2*mesh.topology[i, j]
                dofs[2*j + 1] = 2*mesh.topology[i, j] - 1
            f_global[dofs] += f

    return f_global


def stiffness_matrix(mesh: Mesh, material: Material,
                     plane_stress: bool = True) -> NDArray[np.float64]:
    """Assemble the global stiffness matrix."""

    # assemble material stiffness matrix
    if plane_stress:
        D = material.plane_stress_matrix()
    else:
        D = material.plane_strain_matrix()

    # iterate over all elements to populate global stiffness matrix
    n_dofs = 2*mesh.n_nodes
    K_global = np.zeros(shape=(n_dofs, n_dofs), dtype=np.float64)
    for i in range(mesh.n_elements):

        # construct stiffness matrix for element "i" by performing
        # Gauss quadrature
        K = np.zeros(shape=(8, 8), dtype=np.float64)
        x = mesh.coordinates[mesh.topology[i]]
        for j in range(4):
            xi, eta = Q4_GAUSS_POINTS[j]
            w = Q4_GAUSS_WEIGHTS[j]
            dphi = q4gradient(xi=xi, eta=eta)
            jac = dphi @ x
            det = np.linalg.det(a=jac)
            if det <= 0.0:
                raise ValueError('Determinant of Jacobian is <= 0')
            dphi_dx = np.linalg.solve(a=jac, b=dphi)
            B = np.zeros(shape=(3, 8), dtype=np.float64)
            for k in range(4):
                B[0, 2*k] = B[2, 2*k + 1] = dphi_dx[0, k]
                B[1, 2*k + 1] = B[2, 2*k] = dphi_dx[1, k]
            K += w*(B.T @ D @ B)*det

        # pack the global stiffness matrix
        dofs = np.zeros(shape=8, dtype=np.int32)
        for j in range(4):
            dofs[2*j] = 2*mesh.topology[i, j]
            dofs[2*j + 1] = 2*mesh.topology[i, j] + 1
        for j in range(8):
            for k in range(8):
                K_global[dofs[j], dofs[k]] += K[j, k]

    return K_global

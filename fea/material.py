#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tools for engineering materials."""

import numpy as np
from numpy.typing import NDArray


class Material:
    """Container for a linear elastic material."""
    E: float  # Young's modulus
    nu: float  # Poisson's ratio

    def __init__(self, E: float = 200e9, nu: float = 0.28) -> None:
        self.E = float(E)
        self.nu = float(nu)

    def plane_strain_matrix(self) -> NDArray[np.float64]:
        """Generate the plane strain stiffness matrix."""
        D = np.zeros(shape=(3, 3), dtype=np.float64)
        G = self.E/(2.0*(1.0 + self.nu))
        lame = self.E*self.nu/((1.0 + self.nu)*(1.0 - 2.0*self.nu))
        D[0, 0] = D[1, 1] = 2.0*G + lame
        D[0, 1] = D[1, 0] = lame
        D[2, 2] = G
        return D

    def plane_stress_matrix(self) -> NDArray[np.float64]:
        """Generate the plane stress stiffness matrix."""
        D = np.zeros(shape=(3, 3), dtype=np.float64)
        tmp = self.E/(1.0 - self.nu**2)
        D[0, 0] = D[1, 1] = tmp
        D[0, 1] = D[1, 0] = tmp*self.nu
        D[2, 2] = tmp*(1.0 + self.nu)
        return D

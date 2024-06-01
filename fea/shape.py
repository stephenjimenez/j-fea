#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finite element shape functions."""

import numpy as np
from numpy.typing import NDArray


Q4_GAUSS_POINTS = [[-0.5773502691896257, -0.5773502691896257],
                   [+0.5773502691896257, -0.5773502691896257],
                   [+0.5773502691896257, +0.5773502691896257],
                   [-0.5773502691896257, +0.5773502691896257]]

Q4_GAUSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]


def q4gradient(xi: float, eta: float) -> NDArray[np.float64]:
    """Shape function gradients for a Q4 element."""
    dphi = np.empty(shape=(2, 4), dtype=np.float64)
    dphi[0, 0] = -0.25*(1.0 - eta)
    dphi[0, 1] = +0.25*(1.0 - eta)
    dphi[0, 2] = +0.25*(1.0 + eta)
    dphi[0, 3] = -0.25*(1.0 + eta)
    dphi[1, 0] = -0.25*(1.0 - xi)
    dphi[1, 1] = -0.25*(1.0 + xi)
    dphi[1, 2] = +0.25*(1.0 + xi)
    dphi[1, 3] = +0.25*(1.0 - xi)
    return dphi


def q4shape(xi: float, eta: float) -> NDArray[np.float64]:
    """Shape functions for a Q4 element."""
    phi = np.empty(shape=4, dtype=np.float64)
    phi[0] = 0.25*(1.0 - xi)*(1.0 - eta)
    phi[1] = 0.25*(1.0 + xi)*(1.0 - eta)
    phi[2] = 0.25*(1.0 + xi)*(1.0 + eta)
    phi[3] = 0.25*(1.0 - xi)*(1.0 + eta)
    return phi

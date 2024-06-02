#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tools for managing the finite element mesh."""

import numpy as np
from numpy.typing import NDArray


class Mesh:
    """Container for the finite element mesh."""
    coordinates: NDArray[np.float64]
    topology: NDArray[np.int32]

    def __init__(self, coordinates: NDArray[np.float64],
                 topology: NDArray[np.int32]) -> None:
        self.coordinates = coordinates
        self.topology = topology

    def boundary(self) -> NDArray[np.int32]:
        """Generate an array of nodes existing on the mesh boundary."""
        edges = {}
        for element in self.topology:
            for i in range(4):
                a, b = element[i], element[(i + 1) % 4]
                key = (a, b) if a < b else (b, a)
                if key in edges:
                    edges[key] += 1
                else:
                    edges[key] = 1
        s = set()
        for key, val in edges.items():
            if val == 1:
                s.update(key)
        nodes = np.array(object=list(s), dtype=np.int32)
        return np.unique(a=nodes)

    @staticmethod
    def grid_mesh(m: int, n: int, L: float = 1.0, H: float = 1.0) -> 'Mesh':
        """Generate an m-by-n grid mesh."""
        topology = np.empty(shape=(m*n, 4), dtype=np.int32)
        k = 0
        for i in range(n):
            for j in range(m):
                topology[k, 0] = i*(m + 1) + j
                topology[k, 1] = i*(m + 1) + j + 1
                topology[k, 2] = (i + 1)*(m + 1) + j + 1
                topology[k, 3] = (i + 1)*(m + 1) + j
                k += 1
        coordinates = np.empty(shape=((m + 1)*(n + 1), 2), dtype=np.float64)
        dx = L/m
        dy = H/n
        k = 0
        for i in range(n + 1):
            for j in range(m + 1):
                coordinates[k, 0] = j*dx
                coordinates[k, 1] = i*dy
                k += 1
        return Mesh(coordinates=coordinates, topology=topology)

    @property
    def n_elements(self) -> int:
        return self.topology.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.coordinates.shape[0]

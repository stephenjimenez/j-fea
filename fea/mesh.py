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

    def external_edges(self) -> NDArray[np.int32]:
        """Generate an array of edges existing on the mesh boundary."""
        edges = {}
        for element in self.topology:
            for i in range(4):
                a, b = element[i], element[(i + 1) % 4]
                key = (a, b) if a < b else (b, a)
                if key in edges:
                    edges[key] += 1
                else:
                    edges[key] = 1

    @property
    def n_elements(self) -> int:
        return self.topology.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.coordinates.shape[0]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for finite element shape functions."""

import unittest

import numpy as np

import fea.shape


class TestShape(unittest.TestCase):
    """Unit tests for finite element shape functions."""

    def setUpClass() -> None:
        """Run once at the beginning."""
        np.random.seed(1337)

    def test_interpolation(self) -> None:
        """Test the interpolating capability """
        x = np.array(
            object=[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            dtype=np.float64)
        u = 2.0*x[:, 0] - 3.3*x[:, 1]
        xi, eta = 1.0 - 2.0*np.random.random(2)
        phi = fea.shape.q4shape(xi=xi, eta=eta)
        dphi = fea.shape.q4gradient(xi=xi, eta=eta)
        self.assertAlmostEqual(first=(phi @ u), second=(2.0*xi - 3.3*eta))
        self.assertTrue(np.allclose(a=(dphi @ u), b=[2.0, -3.3]))

    def test_partition_of_unit(self) -> None:
        """Ensure that the shape functions satisfy the partition of
        unity quality.
        """
        xi, eta = 1.0 - 2.0*np.random.random(2)
        phi = fea.shape.q4shape(xi=xi, eta=eta)
        dphi = fea.shape.q4gradient(xi=xi, eta=eta)
        self.assertAlmostEqual(first=phi.sum(), second=1.0)
        self.assertTrue(np.allclose(a=dphi.sum(axis=1), b=0.0))


if __name__ == '__main__':
    unittest.main()

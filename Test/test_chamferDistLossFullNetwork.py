from unittest import TestCase
import numpy as np
import numpy.testing as testing

from loss_function import ChamferDistLossFullNetwork


class TestChamferDistLossFullNetwork(TestCase):
    def setUp(self):
        self.loss_fn = ChamferDistLossFullNetwork()

    def test_forward(self):
        self.fail()

    def test_chamfer_dist(self):
        self.fail()

    def test_get_closest_points(self):
        self.fail()

    def test_get_distances_tensor(self):
        self.fail()

    def test_get_closest_points_tensor(self):
        self.fail()

    def test_chamfer_dist_tensor(self):
        self.fail()

    def test_to_string(self):
        self.fail()

    def test_from_string(self):
        self.fail()


class TestGetDistances(TestChamferDistLossFullNetwork):
    def setUp(self):
        super().setUp()
        self.cloud1 = np.array([
            [1, 2, 3],
            [7, 8, 9],
            [4, 5, 6]
        ])
        self.cloud2 = np.array([
            [3, 2, 1],
            [6, 5, 4],
            [9, 8, 7]
        ])

    def test_get_distances(self):
        testing.assert_almost_equal(
            self.loss_fn.get_distances(
                self.cloud1,
                self.cloud2
            ),
            np.array([
                [8, 35, 116],
                [116, 35, 8],
                [35, 8, 35]
            ])
        )
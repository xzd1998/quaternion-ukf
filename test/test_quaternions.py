import unittest

import numpy as np

from estimator.quaternions import Quaternions, ZeroQuaternionException


class QuaternionsTest(unittest.TestCase):

    q_rot_all = Quaternions([0, 1, 1, 1])
    q_identity = Quaternions([1, 0, 0, 0])
    q_rot_x = Quaternions([0, 1, 0, 0])
    q_rot_y = Quaternions([0, 0, 1, 0])
    q_rot_z = Quaternions([0, 0, 0, 1])

    epsilon = 1e-5

    @staticmethod
    def get_random_qs(n):
        qs = []
        for i in range(n):
            np.random.seed(i)
            qs.append(Quaternions(np.random.randn(Quaternions.ndim)))
        return tuple(qs)

    def test_zero_quaternion_input(self):
        invalid = np.zeros(Quaternions.ndim)
        self.assertRaises(ZeroQuaternionException, Quaternions, invalid)

    def test_invalid_dim_input(self):
        invalid = np.ones(5)
        self.assertRaises(ValueError, Quaternions, invalid)

    def test_equality_singleton(self):
        q_rot_x_neg = Quaternions([0, -1, 0, 0])
        self.assertEqual(q_rot_x_neg, QuaternionsTest.q_rot_x)

    def test_equality_array(self):
        q_rot_x_neg = Quaternions([0, -1, 0, 0])
        qs1 = Quaternions.from_quaternions(q_rot_x_neg, QuaternionsTest.q_rot_x)
        qs2 = Quaternions.from_quaternions(QuaternionsTest.q_rot_x, q_rot_x_neg)
        self.assertEqual(qs1, qs2)

    def test_mult_inverse_singleton(self):
        q = QuaternionsTest.get_random_qs(1)[0]
        self.assertEqual(q.q_multiply(q.inverse()), QuaternionsTest.q_identity)

    def test_mult_inverse_array(self):
        qlist = QuaternionsTest.get_random_qs(2)
        qs = Quaternions.from_quaternions(*qlist)
        qs_identity = Quaternions.from_quaternions(QuaternionsTest.q_identity, QuaternionsTest.q_identity)
        self.assertEqual(qs.q_multiply(qs.inverse()), qs_identity)

    def test_mult_associativity(self):
        qlist = QuaternionsTest.get_random_qs(3)
        self.assertEqual(
            qlist[0].q_multiply(qlist[1].q_multiply(qlist[2])),
            qlist[0].q_multiply(qlist[1]).q_multiply(qlist[2])
        )

    def test_round_trip_vector(self):
        qlist = QuaternionsTest.get_random_qs(2)
        qs = Quaternions.from_quaternions(*qlist)
        self.assertEqual(Quaternions.from_vectors(qs.to_vectors()), qs)

    def test_zero_vector(self):
        self.assertEqual(Quaternions.from_vectors(np.zeros(3)), QuaternionsTest.q_identity)

    def test_find_mean_at_pi(self):
        q_rot_x_neg = Quaternions([0, -1, 0, 0])
        q_0 = QuaternionsTest.get_random_qs(1)[0]
        qs = Quaternions.from_quaternions(q_rot_x_neg, QuaternionsTest.q_rot_x)
        self.assertEqual(qs.find_q_mean(q_0), q_rot_x_neg)

    def test_find_mean_near_pi(self):
        dist = 0.1
        q_near = Quaternions.from_vectors([-np.pi + dist, 0, 0])
        q_pi = Quaternions.from_vectors([np.pi, 0, 0])
        q_expected = Quaternions.from_vectors([-np.pi + dist / 2, 0, 0])
        q_0 = QuaternionsTest.get_random_qs(1)[0]
        qs = Quaternions.from_quaternions(q_near, q_pi)
        self.assertEqual(qs.find_q_mean(q_0), q_expected)

"""
Filename: quaternion.py
Author: Matt Lisle
Date: 02/10/19
Description: Quaternion toolbox for UKF
"""

import numpy as np


class Quaternion(object):
    DIMENSIONS = 4

    def __init__(self, q):
        self.q = np.array(q)

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, array):
        if array.shape[0] is not Quaternion.DIMENSIONS:
            raise ValueError("Invalid number of dimensions {}".format(q.shape[0]))
        self._q = array

    def find_q_mean(self, q_mean, iterations=100):
        """
        Finds the mean of a quaternion or array of quaternions using gradient descent
        :param iterations: max number of iterations
        :param q_mean: initial mean guess
        :return: final mean
        """
        if np.all(q_mean.q == 0):
            raise ValueError("Cannot find mean starting at zero quaternion")

        for i in range(iterations):
            err_quat = self.q_multiply(q_mean.inverse())  # Equation 52
            err_rot = err_quat.to_vector()
            err_rot_mean = np.mean(err_rot, axis=1)  # Equation 54
            err_quat_mean = Quaternion.from_vector(err_rot_mean)
            q_mean = err_quat_mean.q_multiply(q_mean)  # Equation 55
            if np.linalg.norm(err_rot_mean) < 1e-5:
                break

        return q_mean

    def q_multiply(self, q2):
        """
        Multiplies two quaternions (or arrays of quaternions) together taking advantage of numpy broadcasting
        :param q2: quaternion to multiply with
        :return: resulting quaternion or array of quaternions
        """
        if len(self.q.shape) > len(q2.q.shape):
            result = np.zeros(self.q.shape)
        else:
            result = np.zeros(q2.q.shape)
        result[0] = self.q[0] * q2.q[0] - self.q[1] * q2.q[1] - self.q[2] * q2.q[2] - self.q[3] * q2.q[3]
        result[1] = self.q[0] * q2.q[1] + self.q[1] * q2.q[0] - self.q[2] * q2.q[3] + self.q[3] * q2.q[2]
        result[2] = self.q[0] * q2.q[2] + self.q[1] * q2.q[3] + self.q[2] * q2.q[0] - self.q[3] * q2.q[1]
        result[3] = self.q[0] * q2.q[3] - self.q[1] * q2.q[2] + self.q[2] * q2.q[1] + self.q[3] * q2.q[0]
        return Quaternion(result.astype(float) / np.linalg.norm(result, axis=0))

    def to_vector(self):
        """
        Converts a quaternion or array of quaternions to a vector or array of vectors
        :return: vector(s) as a numpy array
        """
        theta = 2 * np.arccos(self.q[0])
        if len(self.q.shape) == 1:
            v = np.zeros(3)
        else:
            v = np.zeros((3, self.q.shape[-1]))
        if np.any(theta == 0):
            ind = theta == 0
            not_ind = theta != 0
            v[..., ind] = np.zeros((3, np.sum(ind)))
            v[..., not_ind] = theta[not_ind].astype(float) / np.sin(theta[not_ind] * .5) * self.q[1:, not_ind]
            return v
        v = theta.astype(float) / np.sin(theta * .5) * self.q[1:]
        return v.astype(float) / np.linalg.norm(v, axis=0)

    def inverse(self):
        """
        :return: the inverse of a quaternion or array of quaternions
        """
        qinv = np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]]).astype(float)
        qinv /= np.linalg.norm(qinv)
        return Quaternion(qinv)

    def to_rotation_matrix(self):
        """
        Converts quaternion or array of quaternions to a rotation matrix or matrices
        :return: rotation matrix or matrices as numpy arrays
        """
        u0 = self.q[0]
        u = self.q[1:].reshape(3, 1)
        uhat = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        R = (u0 ** 2 - np.matmul(u.T, u)) * np.identity(3) + 2 * u0 * uhat + 2 * np.matmul(u, u.T)

        return R.T

    @staticmethod
    def from_vector(v):
        """
        Converts vector or array of vectors to a quaternion or array of quaternions
        :param v: vector to convert
        :return: resulting quaternion(s)
        """
        a = np.linalg.norm(v, axis=0)
        b = np.zeros(v.shape)
        if np.any(a == 0):
            ind = a == 0
            not_ind = a != 0
            b[:, ind] = np.zeros((3, np.sum(ind)))
            b[:, not_ind] = v[:, not_ind].astype(float) / a[not_ind]
        else:
            b = v.astype(float) / a
        q = np.array([np.cos(a * .5), b[0] * np.sin(a * .5), b[1] * np.sin(a * .5), b[2] * np.sin(a * .5)])
        return Quaternion(q.astype(float) / np.linalg.norm(q, axis=0))

    @staticmethod
    def group(*qs):
        return Quaternion(np.array([q.q for q in qs]).T)

    def __eq__(self, other):
        return self.q == other.q

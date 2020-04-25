"""
Filename: quaternion.py
Author: Matt Lisle
Date: 02/10/19
Description: Quaternion toolbox for UKF
"""

import numpy as np
from warnings import warn


class QuaternionArray(object):
    DIMENSIONS = 4

    def __init__(self, q):
        self.q = np.array(q) / np.linalg.norm(q, axis=0)

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, array):
        if array.shape[0] is not QuaternionArray.DIMENSIONS:
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

        i = 0
        for _ in range(iterations):
            err_quat = self.q_multiply(q_mean.inverse())  # Equation 52
            err_quat = QuaternionArray(err_quat.q * np.sign(err_quat.q[0]))
            err_rot = err_quat.to_vector()
            err_rot_mean = np.mean(err_rot, axis=1)  # Equation 54
            err_quat_mean = QuaternionArray.from_vector(err_rot_mean)
            q_mean = err_quat_mean.q_multiply(q_mean)  # Equation 55

            err = np.linalg.norm(err_rot_mean)
            if err < 1e-5:
                break
            i += 1

        if i == iterations - 1:
            warn("Reached max number of iterations to find mean")

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
        return QuaternionArray(result.astype(float) / np.linalg.norm(result, axis=0))

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
        return v.astype(float)  # / np.linalg.norm(v, axis=0)

    def inverse(self):
        """
        :return: the inverse of a quaternion or array of quaternions
        """
        qinv = np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]]).astype(float)
        qinv /= np.linalg.norm(qinv)
        return QuaternionArray(qinv)

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
        return QuaternionArray(q.astype(float) / np.linalg.norm(q, axis=0))

    @staticmethod
    def group(*qs):
        return QuaternionArray(np.array([q.q for q in qs]).T)

    def __eq__(self, other):
        return self.q == other.q

    def __repr__(self):
        return self.__str__()

    def __str__(self):

        n_digits = 5

        def get_elem_str(elem):
            return str(round(elem, n_digits))

        characters = [c for c in __class__.__name__]
        characters.append('(')
        characters.append('\n')
        quaternions = self.q.reshape(QuaternionArray.DIMENSIONS, -1)
        axes = ['w', 'i', 'j', 'k']

        max_chars = [0, 0, 0, 0]
        for row in range(quaternions.shape[0]):
            for col in range(quaternions.shape[1]):

                n_chars = len(get_elem_str(quaternions[row, col]))

                if n_chars > max_chars[row]:
                    max_chars[row] = n_chars

        for idx in range(quaternions.shape[1]):
            characters.append(' ')

            row = 0
            for (element, axis) in zip(quaternions[:, idx], axes):
                elem_str = get_elem_str(element)

                for _ in range(max_chars[row] - len(elem_str) + 1):
                    characters.append(' ')

                characters += [c for c in get_elem_str(element)]
                characters += axis
                row += 1

            characters.append('\n')

        characters.append(')')

        return "".join(characters)

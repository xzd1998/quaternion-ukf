"""
Quaternions wraps numpy arrays to make operations on arrays of quaternions easier.
Because the quaternion UKF has 15 sigma points to project forward and because we
want to take advantage of numpy broadcasting, the array wrapped by Quaternions is
of shape (4, N) where N >= 0.

So, Quaternions can either represent a single quaternion or multiple, and operations
are allowed between two Quaternions objects when:

* Their array shapes match, so all operations are one-to-one
* One is a single quaternion and the other is not, so all operations are one-to-many

For example all of the following are allowed:

.. code-block::
   :linenos:

   import numpy as np
   from estimator.quaternions import Quaternions

   single = Quaternions.from_vectors(np.zeros(3))
   multiple = Quaternions.from_vectors(np.ones((3, 4)))

   # All of the below are allowed
   print(single.q_multiply(single))
   print(single.q_multiply(multiple))
   print(multiple.q_multiply(single))
   print(multiple.q_multiply(multiple))
"""

import numpy as np


class Quaternions:
    """
    Wrapper for numpy arrays for quaternion math

    :cvar NDIM: size of dimension 0 of the wrapped array
    :cvar EPSILON: tolerance for finding the mean of multiple quaternions
    """

    NDIM = 4
    EPSILON = 1e-5

    def __init__(self, array):
        """
        Wraps the numpy array in the Quaternion class

        :param array: (4,N) numpy array where N >= 0
        """
        self.array = array / np.linalg.norm(array, axis=0)

    @property
    def array(self):
        """Getter for wrapped array"""
        return self._array

    @array.setter
    def array(self, array):
        """Setter for wrapped array"""
        if len(array) is not Quaternions.NDIM:
            raise ValueError("Invalid number of dimensions {}".format(array.shape[0]))
        self._array = array

    @property
    def length(self):
        """Number of quaternions in this Quaternions object"""
        if len(self.array.shape) > 1:
            return self.array.shape[1]
        return 1

    def with_positive_scalar(self):
        """
        Makes the sign of the scalar component positive while representing the same rotation
        """
        self.array *= np.sign(self.array[0])

    def find_q_mean(self, q_mean, iterations=1000):
        """
        Finds the mean of a quaternion or array of quaternions using gradient descent

        :param iterations: max number of iterations
        :param q_mean: initial mean guess
        :return: final mean
        """
        if np.all(q_mean.array == 0):
            raise ValueError("Cannot find mean starting at zero quaternion")

        for _ in range(iterations):
            err_quat = self.q_multiply(q_mean.inverse())  # Equation 52
            err_quat.with_positive_scalar()

            err_rot = err_quat.to_vectors()
            err_rot_mean = np.mean(err_rot, axis=1)  # Equation 54

            err_quat_mean = Quaternions.from_vectors(err_rot_mean)
            q_mean = err_quat_mean.q_multiply(q_mean)  # Equation 55

            err = np.linalg.norm(err_rot_mean)
            if err < Quaternions.EPSILON:
                break

        return q_mean

    def q_multiply(self, that):
        """
        Multiplies two quaternions (or arrays of quaternions) together taking advantage of
        numpy broadcasting

        :param that: quaternion to multiply with
        :return: resulting quaternion or array of quaternions
        """

        if len(self.array.shape) > len(that.array.shape):
            result = np.zeros(self.array.shape)
        else:
            result = np.zeros(that.array.shape)

        result[0] = self.array[0] * that.array[0] - self.array[1] * that.array[1] - \
                    self.array[2] * that.array[2] - self.array[3] * that.array[3]
        result[1] = self.array[0] * that.array[1] + self.array[1] * that.array[0] - \
                    self.array[2] * that.array[3] + self.array[3] * that.array[2]
        result[2] = self.array[0] * that.array[2] + self.array[1] * that.array[3] + \
                    self.array[2] * that.array[0] - self.array[3] * that.array[1]
        result[3] = self.array[0] * that.array[3] - self.array[1] * that.array[2] + \
                    self.array[2] * that.array[1] + self.array[3] * that.array[0]

        return Quaternions(result.astype(float) / np.linalg.norm(result, axis=0))

    def to_vectors(self):
        """
        Converts a quaternion or array of quaternions to a vector or array of vectors

        :return: vector(s) as a numpy array
        """
        theta = 2 * np.arccos(self.array[0])
        if len(self.array.shape) == 1:
            vectors = np.zeros(3)
        else:
            vectors = np.zeros((3, self.array.shape[-1]))
        if np.any(theta == 0):
            ind = theta == 0
            not_ind = ~ind
            vectors[..., ind] = np.zeros((3, np.sum(ind)))
            vectors[..., not_ind] = (theta[not_ind] / np.sin(theta[not_ind] * .5) *
                                     self.array[1:, not_ind])
        else:
            vectors = theta / np.sin(theta * .5) * self.array[1:]
        vectors[vectors >= np.pi] -= 2 * np.pi
        vectors[vectors < -np.pi] += 2 * np.pi
        return vectors

    def rotate_vector(self, vector):
        """
        Rotates a vector with this array of n quaternions

        :param vector: (3,) numpy array to rotate
        :return: (3, n) matrix of vectors if n > 1, otherwise a (3,) vector
        """

        dim0, *dims = vector.shape
        if dim0 != 3 and not all(dims == 1):
            raise ValueError("Can't rotate a vector of shape: {}".format(vector.shape))

        quat = Quaternions(np.insert(vector.reshape(-1), 0, 0))
        return self.q_multiply(quat).q_multiply(self.inverse()).array[1:]

    def inverse(self):
        """
        :return: the inverse of a quaternion or array of quaternions
        """
        qinv = np.array([self.array[0], -self.array[1], -self.array[2], -self.array[3]])
        return Quaternions(qinv)

    def to_rotation_matrix(self):
        """
        Converts quaternion or array of quaternions to a rotation matrix or matrices

        :return: rotation matrix or matrices as numpy arrays
        """
        scalar = self.array[0]
        vector = self.array[1:].reshape(3, 1)
        skew = np.array(
            [[0, -vector[2, 0], vector[1, 0]],
             [vector[2, 0], 0, -vector[0, 0]],
             [-vector[1, 0], vector[0, 0], 0]]
        )
        rot = ((scalar ** 2 - np.matmul(vector.T, vector)) * np.identity(3) +
               2 * scalar * skew + 2 * np.matmul(vector, vector.T))

        return rot

    @staticmethod
    def from_vectors(vectors):
        """
        Converts vector or array of vectors to an array of quaternions

        :param vectors: vector to convert
        :return: resulting quaternion(s)
        """
        alpha = np.linalg.norm(vectors, axis=0)
        unit = np.zeros(vectors.shape)
        if np.any(alpha == 0):
            ind = alpha == 0
            not_ind = alpha != 0
            unit[:, ind] = np.zeros((3, np.sum(ind)))
            unit[:, not_ind] = vectors[:, not_ind] / alpha[not_ind]
        else:
            unit = vectors / alpha

        quat = np.array([
            np.cos(alpha / 2),
            unit[0] * np.sin(alpha / 2),
            unit[1] * np.sin(alpha / 2),
            unit[2] * np.sin(alpha / 2)
        ])
        return Quaternions(quat / np.linalg.norm(quat, axis=0))

    @classmethod
    def from_quaternions(cls, *qs):
        """
        Constructs quaternions from list of singleton quaternion arrays
        """
        return cls(np.array([q.array for q in qs]).T)

    @classmethod
    def from_list(cls, quat_list):
        """
        Constructs quaternions from list of 4-element lists of numbers
        """
        return cls(np.array(quat_list).T)

    def __len__(self):
        return self.length

    def __eq__(self, other):
        if self.array.shape != other.array.shape:
            raise ValueError("Cannot compare equality of quaternion arrays with different shapes")
        tol = Quaternions.EPSILON * 10
        result = np.all(np.logical_or(
            np.all(self.array - other.array < tol, axis=0),
            np.all(self.array + other.array < tol, axis=0)
        ))
        return result

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        is_int = isinstance(item, int)
        if is_int and not -self.length <= item < self.length:
            raise IndexError(
                "Index {} out of bounds for {} of length {}"
                .format(item, __class__.__name__, self.length)
            )
        if is_int:
            return Quaternions(self.array[:, item])
        raise TypeError("{} may be indexed only with integers".format(__class__.__name__))

    def __str__(self):

        n_digits = 5

        def get_elem_str(elem):
            return str(round(elem, n_digits))

        characters = list(__class__.__name__)
        characters.append('(')
        characters.append('\n')
        quaternions = self.array.reshape(Quaternions.NDIM, -1)
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

                characters += list(elem_str)
                characters += axis
                row += 1

            characters.append('\n')

        characters.append(')')

        return "".join(characters)

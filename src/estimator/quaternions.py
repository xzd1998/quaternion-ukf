import numpy as np


class Quaternions:
    ndim = 4
    epsilon = 1e-5

    def __init__(self, raw):
        array = np.array(raw)

        norm = np.linalg.norm(array, axis=0)
        if np.any(norm == 0):
            raise ZeroQuaternionException("Found zero quaternion in array: {}".format(array))

        self.array = array / np.linalg.norm(array, axis=0)

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, array):
        if array.shape[0] is not Quaternions.ndim:
            raise ValueError("Invalid number of dimensions {}".format(array.shape[0]))
        self._array = array

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
            err_quat = Quaternions(err_quat.array * np.sign(err_quat.array[0]))

            err_rot = err_quat.to_vectors()
            err_rot_mean = np.mean(err_rot, axis=1)  # Equation 54

            err_quat_mean = Quaternions.from_vectors(err_rot_mean)
            q_mean = err_quat_mean.q_multiply(q_mean)  # Equation 55

            err = np.linalg.norm(err_rot_mean)
            if err < Quaternions.epsilon:
                break

        return q_mean

    def q_multiply(self, q2):
        """
        Multiplies two quaternions (or arrays of quaternions) together taking advantage of numpy broadcasting
        :param q2: quaternion to multiply with
        :return: resulting quaternion or array of quaternions
        """
        if len(self.array.shape) > len(q2.array.shape):
            result = np.zeros(self.array.shape)
        else:
            result = np.zeros(q2.array.shape)

        result[0] = self.array[0] * q2.array[0] - self.array[1] * q2.array[1] - \
                    self.array[2] * q2.array[2] - self.array[3] * q2.array[3]
        result[1] = self.array[0] * q2.array[1] + self.array[1] * q2.array[0] - \
                    self.array[2] * q2.array[3] + self.array[3] * q2.array[2]
        result[2] = self.array[0] * q2.array[2] + self.array[1] * q2.array[3] + \
                    self.array[2] * q2.array[0] - self.array[3] * q2.array[1]
        result[3] = self.array[0] * q2.array[3] - self.array[1] * q2.array[2] + \
                    self.array[2] * q2.array[1] + self.array[3] * q2.array[0]

        return Quaternions(result.astype(float) / np.linalg.norm(result, axis=0))

    def to_vectors(self):
        """
        Converts a quaternion or array of quaternions to a vector or array of vectors
        :return: vector(s) as a numpy array
        """
        theta = 2 * np.arccos(self.array[0])
        if len(self.array.shape) == 1:
            v = np.zeros(3)
        else:
            v = np.zeros((3, self.array.shape[-1]))
        if np.any(theta == 0):
            ind = theta == 0
            not_ind = theta != 0
            v[..., ind] = np.zeros((3, np.sum(ind)))
            v[..., not_ind] = theta[not_ind].astype(float) / np.sin(theta[not_ind] * .5) * self.array[1:, not_ind]
        else:
            v = theta.astype(float) / np.sin(theta * .5) * self.array[1:]
        v[v > np.pi] -= 2 * np.pi
        v[v < -np.pi] += 2 * np.pi
        return v

    def rotate_vector(self, vector):
        """
        Rotates a vector with this array of n quaternions
        :param vector: (3,) numpy array to rotate
        :return: (3, n) matrix of vectors if n > 1, otherwise a (3,) vector
        """
        dim0, *dims = vector.shape
        if dim0 != 3 and not all(dims == 1):
            raise ValueError("Can't rotate a vector of shape: {}".format(vector.shape))
        q = Quaternions(np.insert(vector.reshape(-1), 0, 0))
        return self.q_multiply(q).q_multiply(self.inverse()).array[1:]

    def inverse(self):
        """
        :return: the inverse of a quaternion or array of quaternions
        """
        qinv = np.array([self.array[0], -self.array[1], -self.array[2], -self.array[3]]).astype(float)
        return Quaternions(qinv)

    def to_rotation_matrix(self):
        """
        Converts quaternion or array of quaternions to a rotation matrix or matrices
        :return: rotation matrix or matrices as numpy arrays
        """
        u0 = self.array[0]
        u = self.array[1:].reshape(3, 1)
        uhat = np.array([[0, -u[2, 0], u[1, 0]], [u[2, 0], 0, -u[0, 0]], [-u[1, 0], u[0, 0], 0]])
        R = (u0 ** 2 - np.matmul(u.T, u)) * np.identity(3) + 2 * u0 * uhat + 2 * np.matmul(u, u.T)

        return R

    @staticmethod
    def from_vectors(raw):
        """
        Converts vector or array of vectors to an array of quaternions
        :param raw: vector to convert
        :return: resulting quaternion(s)
        """
        v = np.array(raw)
        a = np.linalg.norm(v, axis=0)
        b = np.zeros(v.shape)
        if np.any(a == 0):
            ind = a == 0
            not_ind = a != 0
            b[:, ind] = np.zeros((3, np.sum(ind)))
            b[:, not_ind] = v[:, not_ind].astype(float) / a[not_ind]
        else:
            b = v.astype(float) / a
        q = np.array([np.cos(a / 2), b[0] * np.sin(a / 2), b[1] * np.sin(a / 2), b[2] * np.sin(a / 2)])
        return Quaternions(q.astype(float) / np.linalg.norm(q, axis=0))

    @classmethod
    def from_quaternions(cls, *qs):
        """
        Constructs quaternions from list of singleton quaternion arrays
        """
        return cls(np.array([q.array for q in qs]).T)

    @classmethod
    def from_list(cls, qs):
        """
        Constructs quaternions from list of 4-element lists of numbers
        """
        return cls(np.array(qs).T)

    def __eq__(self, other):
        if self.array.shape != other.array.shape:
            raise ValueError("Cannot compare equality of quaternion arrays with different shapes")
        tol = Quaternions.epsilon * 10
        result = np.all(np.logical_or(
            np.all(self.array - other.array < tol, axis=0),
            np.all(self.array + other.array < tol, axis=0)
        ))
        return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):

        n_digits = 5

        def get_elem_str(elem):
            return str(round(elem, n_digits))

        characters = [c for c in __class__.__name__]
        characters.append('(')
        characters.append('\n')
        quaternions = self.array.reshape(Quaternions.ndim, -1)
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


class ZeroQuaternionException(Exception):
    def __init__(self, message):
        super().__init__(message)

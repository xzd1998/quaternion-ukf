"""
Integrates gyro data and keeps track of the orientation of the robot
"""

import numpy as np

from estimator.state_estimator import StateEstimator
from estimator.quaternions import Quaternions


class VelocityIntegrator(StateEstimator):
    """
    Considers only gyro data to try to keep track of the orientation of the robot.
    Gyro's are generally quite accurate, but drift in the signal get's integrated,
    exacerbating the estimation error.

    :ivar quats: list of quaternions with the same length as the number of timesteps
    """

    def __init__(self, source):
        super().__init__(source)
        self.quats = []

    def estimate_state(self):
        """
        Uses data provided by the source to estimate the state history of the filter
        After calling this function, the state and rotation history will be defined
        """

        self.rots = np.zeros((3, 3, self.num_data))
        self.quats.append(Quaternions([1, 0, 0, 0]))

        for i in range(0, self.num_data - 1):
            dt = self.ts_imu[i + 1] - self.ts_imu[i]
            self._filter_next(self.vel_data[:, i], dt)

    def _filter_next(self, velocity, dt):
        quat_delta = Quaternions.from_vectors(velocity * dt)
        i = len(self.quats)
        self.quats.append(quat_delta.q_multiply(self.quats[-1]))
        self.rots[..., i] = self.quats[-1].to_rotation_matrix()

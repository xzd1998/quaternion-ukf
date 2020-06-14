import numpy as np

from estimator.data import utilities
from estimator.state_estimator import StateEstimator
from estimator.quaternions import Quaternions


class VectorIntegrator(StateEstimator):

    def __init__(self, source):
        super().__init__(source)
        self.quats = []
        self.acc_calc = np.zeros((3, self.num_data))
        self.g_quat = Quaternions([0, 0, 0, 1])

    def estimate_state(self):
        mu = np.zeros((3, self.num_data))

        for i in range(0, self.num_data - 1):
            dt = self.ts_imu[i + 1] - self.ts_imu[i]
            mu[:, i + 1] = mu[:, i] + self.vel_data[:, i] * dt

        self.rots = utilities.vectors_to_rots(mu)

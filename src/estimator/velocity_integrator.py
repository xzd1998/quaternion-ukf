"""Quaternion Integrator

State estimator which integrates gyro data and keeps track of the state
"""

import numpy as np

from estimator.data import utilities
from estimator.data.datamaker import DataMaker
import estimator.data.trajectoryplanner
from estimator.state_estimator import StateEstimator
from estimator.quaternions import Quaternions


class VelocityIntegrator(StateEstimator):

    state_dof = 6

    def __init__(self, source):
        super().__init__(source)
        self.quats = []
        self.acc_calc = np.zeros((3, self.num_data))
        self.g_quat = Quaternions([0, 0, 0, 1])

    def estimate_state(self):
        self.rots = np.zeros((3, 3, self.num_data))
        self._store_next(Quaternions([1, 0, 0, 0]))

        for i in range(0, self.num_data - 1):
            dt = self.ts_imu[i + 1] - self.ts_imu[i]
            self._filter_next(self.vel_data[:, i], dt)

    def _filter_next(self, w, dt):
        qd = Quaternions.from_vectors(w * dt)
        q_next = qd.q_multiply(self.quats[-1])
        self._store_next(q_next)

    def _store_next(self, q_next):
        i = len(self.quats)
        self.quats.append(q_next)
        self.rots[..., i] = q_next.to_rotation_matrix()
        self.acc_calc[:, i] = q_next.q_multiply(self.g_quat).q_multiply(q_next.inverse()).array[1:]

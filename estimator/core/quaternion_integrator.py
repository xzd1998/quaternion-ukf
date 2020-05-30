import numpy as np

from data import utilities, trajectoryplanner
from data.datamaker import DataMaker
from core.estimator import Estimator
from core.quaternions import Quaternions


class QuaternionIntegrator(Estimator):

    N_DIM = 6

    def __init__(self, source):
        super().__init__(source)
        self.quats = []
        self.acc_calc = np.zeros((3, self.num_data))
        self.g_quat = Quaternions([0, 0, 0, 1])

    def filter_data(self):
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


if __name__ == "__main__":

    planner = trajectoryplanner.round_trip_easy
    source = DataMaker(planner)
    m = np.ones(QuaternionIntegrator.N_DIM)
    b = np.zeros(QuaternionIntegrator.N_DIM)

    f = QuaternionIntegrator(source)
    f.filter_data()

    utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.angles, f.angles)
    # utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.acc_data / g, f.acc_calc)

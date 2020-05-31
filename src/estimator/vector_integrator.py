import numpy as np

from estimator.data import trajectoryplanner, utilities
from estimator.data.datamaker import DataMaker
from estimator.estimator import Estimator
from estimator.quaternions import Quaternions


class VectorIntegrator(Estimator):

    def __init__(self, source):
        super().__init__(source)
        self.quats = []
        self.acc_calc = np.zeros((3, self.num_data))
        self.g_quat = Quaternions([0, 0, 0, 1])

    def filter_data(self):
        mu = np.zeros((3, self.num_data))

        for i in range(0, self.num_data - 1):
            dt = self.ts_imu[i + 1] - self.ts_imu[i]
            mu[:, i + 1] = mu[:, i] + self.vel_data[:, i] * dt

        self.rots = utilities.vectors_to_rots(mu)


if __name__ == "__main__":

    planner = trajectoryplanner.round_trip_easy
    source = DataMaker(planner)

    f = VectorIntegrator(source)
    f.filter_data()

    utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.angles, f.angles)

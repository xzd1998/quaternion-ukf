import argparse
import matplotlib.pyplot as plt
import numpy as np

from data import utilities
from data.datamaker import DataMaker
from data.datastore import DataStore
from data.trajectoryplanner import SimplePlanner
from imufilter import ImuFilter
from quaternions import Quaternions


class QuaternionIntegrator(ImuFilter):

    N_DIM = 6

    def __init__(self, source):
        super().__init__(source)

    def filter_data(self):
        self.rots = np.zeros((3, 3, self.num_data))
        q = Quaternions([1, 0, 0, 0])
        for i in range(0, self.num_data - 1):
            self.rots[..., i] = q.to_rotation_matrix()
            dt = self.ts_imu[i + 1] - self.ts_imu[i]
            q = self._filter_next(q, self.vel_data[:, i], dt)

        self.rots[..., -1] = q.to_rotation_matrix()

    @staticmethod
    def _filter_next(q, w, dt):
        qd = Quaternions.from_vectors(w * dt)
        return q.q_multiply(qd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--datanum", required=False, help="Number of data file (1 to 3 inclusive)")

    args = vars(parser.parse_args())

    num = args["datanum"]
    if not num:
        planner = SimplePlanner()
        source = DataMaker(planner)
        m = np.ones(QuaternionIntegrator.N_DIM)
        b = np.zeros(QuaternionIntegrator.N_DIM)
    else:
        source = DataStore(dataset_number=num, path_to_data="data")

    f = QuaternionIntegrator(source)
    f.filter_data()

    if not num:
        utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.angles, f.angles)
    else:
        f.plot_comparison(f.rots, f.ts_imu, source.rots_vicon, source.ts_vicon)

import argparse
import numpy as np
import matplotlib.pyplot as plt

from data import utilities
from data.datamaker import DataMaker
from data.datastore import DataStore
from data.trajectoryplanner import SimplePlanner, StationaryPlanner


class NoFilter:

    N_DIM = 6

    def __init__(self, source):

        self.num_data = source.ts_imu.shape[-1]

        self.ts_imu = source.ts_imu
        self.data_imu = source.data_imu

        # Initialize properties
        self.mu = np.zeros((NoFilter.N_DIM + 1, self.num_data))
        self.mu[:, 0] = np.array([1, 0, 0, 0, 0, 0, 0])

        self.rots = np.zeros((3, 3, self.num_data))
        for i in range(self.num_data):
            self.rots[..., i] = np.identity(3)

    @property
    def angles(self):
        return utilities.rots_to_angles(self.rots)

    @property
    def accs(self):
        return self.data_imu[:3]

    @property
    def vels(self):
        return self.data_imu[3:]

    def filter_data(self):

        roll, pitch = utilities.accs_to_roll_pitch(self.accs)
        yaw = np.zeros(self.num_data)
        yaw[1:] = self.estimate_yaw()
        self.rots = utilities.angles_to_rots(roll, pitch, yaw)

    def estimate_yaw(self):
        dts = np.diff(self.ts_imu)
        dy_dts = (self.vels[-1, :-1] + self.vels[-1, 1:]) * dts / 2
        return np.cumsum(dy_dts)

    def plot_comparison(self, rots, ts):

        R = np.copy(rots)
        t_vicon = ts.reshape(-1)
        t0 = min(t_vicon[0], self.ts_imu[0])
        R = R[..., t_vicon > self.ts_imu[0]]
        r = self.rots[..., self.ts_imu > t_vicon[0]]

        labels = ["Roll", "Pitch", "Yaw"]

        a = utilities.rots_to_angles(r)
        angs = utilities.rots_to_angles(R)

        for i in range(3):
            plt.figure(i)
            plt.plot(t_vicon[t_vicon > self.ts_imu[0]] - t0, angs[i])
            plt.plot(self.ts_imu[self.ts_imu > t_vicon[0]] - t0, a[i])
            plt.xlabel("Time [s]")
            plt.ylabel(labels[i] + " Angle [rad]")
            plt.grid(True)
            plt.legend(["Truth", "UKF"])
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--datanum", required=False, help="Number of data file (1 to 3 inclusive)")

    args = vars(parser.parse_args())

    num = args["datanum"]
    if not num:
        planner = SimplePlanner()
        source = DataMaker(planner)
        m = np.ones(NoFilter.N_DIM)
        b = np.zeros(NoFilter.N_DIM)
    else:
        source = DataStore(dataset_number=num, path_to_data="data")

    f = NoFilter(source)
    f.filter_data()

    if not num:
        utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.angs_vicon, f.angles)
    else:
        f.plot_comparison(source.rots_vicon, source.ts_vicon)

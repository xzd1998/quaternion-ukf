import argparse
import numpy as np

from data import utilities
from data.datamaker import DataMaker
from data.datastore import DataStore
from data.trajectoryplanner import SimplePlanner
from imufilter import ImuFilter


class NoFilter(ImuFilter):

    N_DIM = 6

    def filter_data(self):

        self.source = source
        roll, pitch = utilities.accs_to_roll_pitch(self.acc_data)
        yaw = np.zeros(self.num_data)
        yaw[1:] = self._estimate_yaw()
        self.rots = utilities.angles_to_rots_zyx(roll, pitch, yaw)

    def _estimate_yaw(self):
        dts = np.diff(self.ts_imu)
        dy_dts = (self.vel_data[-1, :-1] + self.vel_data[-1, 1:]) * dts / 2
        return np.cumsum(dy_dts)


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
        f.plot_comparison(f.rots, f.ts_imu, source.rots_vicon, source.ts_vicon)

import argparse
import matplotlib.pyplot as plt
import numpy as np

from data import utilities
from data.datamaker import DataMaker
from data.datastore import DataStore
from data.trajectoryplanner import SimplePlanner
from imufilter import ImuFilter


class YawIntegrator(ImuFilter):

    N_DIM = 6

    def filter_data(self):

        roll, pitch = utilities.accs_to_roll_pitch(self.acc_data)
        vectors = self._integrate_vel()
        yaw = utilities.rots_to_angles_zyx(utilities.vectors_to_rots(vectors))[-1]
        self.rots = utilities.angles_to_rots_zyx(roll, pitch, yaw)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--datanum", required=False, help="Number of data file (1 to 3 inclusive)")

    args = vars(parser.parse_args())

    num = args["datanum"]
    if not num:
        planner = SimplePlanner()
        source = DataMaker(planner)
        m = np.ones(YawIntegrator.N_DIM)
        b = np.zeros(YawIntegrator.N_DIM)
    else:
        source = DataStore(dataset_number=num, path_to_data="data")

    f = YawIntegrator(source)
    f.filter_data()

    if not num:
        utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.angles, f.angles)
    else:
        f.plot_comparison(f.rots, f.ts_imu, source.rots_vicon, source.ts_vicon)

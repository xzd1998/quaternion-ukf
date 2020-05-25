import argparse
import matplotlib.pyplot as plt
import numpy as np

from data import utilities
from data.datamaker import DataMaker
from data.datastore import DataStore
from data.trajectoryplanner import SimplePlanner
from data import trajectoryplanner
from imufilter import ImuFilter


class RollPitchCalculator(ImuFilter):

    N_DIM = 6

    def filter_data(self):

        roll, pitch = utilities.accs_to_roll_pitch(self.acc_data)
        yaw = np.zeros(roll.shape[0])
        self.rots = utilities.angles_to_rots_zyx(roll, pitch, yaw)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--datanum", required=False, help="Number of data file (1 to 3 inclusive)")

    args = vars(parser.parse_args())

    num = args["datanum"]
    if not num:
        planner = trajectoryplanner.round_trip_easy
        source = DataMaker(planner)
        m = np.ones(RollPitchCalculator.N_DIM)
        b = np.zeros(RollPitchCalculator.N_DIM)
    else:
        source = DataStore(dataset_number=num, path_to_data="data")

    f = RollPitchCalculator(source)
    f.filter_data()

    if not num:
        utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.angles, f.angles)
    else:
        f.plot_comparison(f.rots, f.ts_imu, source.rots_vicon, source.ts_vicon)

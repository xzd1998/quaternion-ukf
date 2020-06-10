"""
Quaternion UKF Demo
^^^^^^^^^^^^^^^^^^^
"""

import argparse

import numpy as np

from estimator.constants import STATE_DOF
from estimator.data.datamaker import DataMaker
from estimator.data.datastore import DataStore
from estimator.data.trajectoryplanner import RoundTripPlanner
from estimator.quaternionukf import QuaternionUkf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-D",
        "--datanum",
        required=False, help="Number of data file (1 to 3 inclusive)"
    )

    args = vars(parser.parse_args())

    # Noise parameters for UKF
    R = np.identity(STATE_DOF) * .1
    Q = np.copy(R)

    num = args["datanum"]
    if not num:
        planner = RoundTripPlanner()
        data_source = DataMaker(planner)
    else:
        data_source = DataStore(dataset_number=num, path_to_data="estimator/data/")

    estimator = QuaternionUkf(data_source, R, Q)
    estimator.estimate_state()

    estimator.plot_comparison(
        estimator.rots,
        estimator.ts_imu,
        data_source.rots_vicon,
        data_source.ts_vicon
    )

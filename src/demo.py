"""
Quaternion UKF Demo
^^^^^^^^^^^^^^^^^^^
"""

import argparse

import numpy as np

from estimator.constants import STATE_DOF
from estimator.data import utilities
from estimator.data.datamaker import DataMaker
from estimator.data.datastore import DataStore
from estimator.data.trajectoryplanner import RoundTripPlanner
from estimator.quaternionukf import QuaternionUkf
from estimator.roll_pitch_calculator import RollPitchCalculator
from estimator.velocity_integrator import VelocityIntegrator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-D",
        "--datanum",
        required=False, help="Number of data file (1 to 3 inclusive)"
    )

    args = vars(parser.parse_args())

    # Noise parameters for UKF
    R = np.identity(STATE_DOF) * .05
    R[4:, 4:] *= 5
    Q = np.copy(R)

    num = args["datanum"]
    if not num:
        planner = RoundTripPlanner()
        data_source = DataMaker(planner)
    else:
        data_source = DataStore(dataset_number=num, path_to_data="estimator/data/")

    ukf = QuaternionUkf(data_source, R, Q)
    ukf.estimate_state()

    integrator = VelocityIntegrator(data_source)
    integrator.estimate_state()

    calculator = RollPitchCalculator(data_source)
    calculator.estimate_state()

    estimators = [ukf, integrator, calculator]
    legend_labels = [est.__class__.__name__ for est in estimators]
    legend_labels.append("Vicon")

    time_vectors = [data_source.ts_imu for _ in estimators]
    time_vectors.append(data_source.ts_vicon)

    angles = [utilities.rots_to_angles_zyx(est.rots) for est in estimators]
    angles.append(utilities.rots_to_angles_zyx(data_source.rots_vicon))

    utilities.plot_data_comparison(
        legend_labels,
        ["Roll", "Pitch", "Yaw"],
        time_vectors,
        angles
    )

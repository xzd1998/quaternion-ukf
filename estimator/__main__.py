"""
Demo script when the module is run
"""

import argparse
import os

import numpy as np

from estimator.constants import STATE_DOF, RCPARAMS, PATH
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
        "-d",
        "--dataset-num",
        nargs="?",
        required=False,
        help="Number associated with dataset (1 to 3 inclusive)"
    )

    args = parser.parse_args()

    R = np.identity(STATE_DOF)
    Q = np.zeros(STATE_DOF)

    if not args.dataset_num:
        planner = RoundTripPlanner(acc_magnitude=0.0005, noise_stddev=0.02, drift_stddev=0.002)
        figure_prefix = planner.__class__.__name__.lower()
        data_source = DataMaker(planner)
        R[:3, :3] *= planner.noise_stddev ** 2
        R[3:, 3:] *= np.var(planner.drift)
        Q = np.copy(R)
    else:
        data_source = DataStore(dataset_number=args.dataset_num, path_to_data="estimator/data/")
        figure_prefix = "dataset_%s" % args.dataset_num
        R *= .05
        Q = np.copy(R) / 2
        R[:3, :3] *= 15

    print("Running estimators...")

    ukf = QuaternionUkf(data_source, R, Q)
    ukf.estimate_state()

    integrator = VelocityIntegrator(data_source)
    integrator.estimate_state()

    calculator = RollPitchCalculator(data_source)
    calculator.estimate_state()

    estimators = [calculator, integrator, ukf]
    legend_labels = [est.__class__.__name__ for est in estimators]
    legend_labels.append("Vicon")

    time_vectors = [data_source.ts_imu for _ in estimators]
    time_vectors.append(data_source.ts_vicon)

    angles = [utilities.rots_to_angles_zyx(est.rots) for est in estimators]
    angles.append(utilities.rots_to_angles_zyx(data_source.rots_vicon))

    print("Results:")

    max_length = max(len(name) for name in legend_labels)
    for estimator in estimators:
        NAME = estimator.__class__.__name__
        rmse_list = ["%.3f" % rmse for rmse in estimator.evaluate_estimation()]
        if NAME == "RollPitchCalculator":
            rmse_list[-1] = "-.---"
        print("  {} RMSE: {}{}"
              .format(NAME, " " * (max_length - len(NAME)), rmse_list)
              .replace("'", ""))

    utilities.plot_data_comparison(legend_labels,
                                   ["Roll", "Pitch", "Yaw"],
                                   time_vectors,
                                   angles,
                                   RCPARAMS,
                                   os.path.join(PATH, figure_prefix))

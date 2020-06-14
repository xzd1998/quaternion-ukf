"""
This demo plots a comparison of the UKF, the result of just integrating the gyro data,
and the result of calculating the roll and pitch from the acceleromter data correctly.
See the README for why the yaw can't be calculated directly.

To run the estimators with toy data, you can use the following:

.. code-block:: bash
   python3 demo.py

To run the estimators on real data from an IMU, specify a dataset number from 1 to 3:

.. code-block:: bash
   python3 demo.py --dataset-num 1

These datasets were provided by the instructors as part of the project. The output of
the demo are the graphs of roll, pitch, and yaw, and the RMSE of each angle is printed
to stdout.

For a well-tuned filter, the RMSE of the UKF will beat (be lower than) that of integrating
the gyro data _or_ calculating roll and pitch from the accelerometer.

This is evident when you look at the results of running the estimators on the toy data,
which has the following output

.. code-block::
   RollPitchCalculator RMSE: [0.0019, 0.0024, 0.9610]
   VelocityIntegrator RMSE:  [0.6236, 0.4494, 0.3113]
   QuaternionUkf RMSE:       [0.0013, 0.0017, 0.1045]

Even though the gyro data has a _lot_ of drift, the UKF is able to fuse the sensor data into
a more accurate estimate than if one sensor had been trusted completely.

Getting a good model of the noise for the toy data is easy because I defined the model, but
the same isn't true for the real IMU data. I've made my best guess for the process and
measurement noise by looking at the coefficient of determination from calibrating the data.
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
        "-d",
        "--dataset-num",
        required=False, help="Number of data file (1 to 3 inclusive)"
    )

    args = vars(parser.parse_args())

    R = np.identity(STATE_DOF)
    Q = np.zeros(STATE_DOF)

    num = args["dataset_num"]
    if not num:
        planner = RoundTripPlanner(acc_magnitude=0.0005, noise_stddev=0.02, drift_stddev=0.002)
        data_source = DataMaker(planner)
        R[:3, :3] *= planner.noise_stddev ** 2
        R[3:, 3:] *= np.var(planner.drift)
        Q = np.copy(R)
    else:
        data_source = DataStore(dataset_number=num, path_to_data="estimator/data/")
        R *= .05
        Q = np.copy(R)
        R[:3, :3] *= 10

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

    max_length = max(len(name) for name in legend_labels)
    for estimator in estimators:
        NAME = estimator.__class__.__name__
        rmse_list = ["%.4f" % rmse for rmse in estimator.evaluate_estimation()]
        print(
            "{} RMSE: {}{}"
            .format(NAME, " " * (max_length - len(NAME)), rmse_list)
            .replace("'", "")
        )

    utilities.plot_data_comparison(
        legend_labels,
        ["Roll", "Pitch", "Yaw"],
        time_vectors,
        angles
    )

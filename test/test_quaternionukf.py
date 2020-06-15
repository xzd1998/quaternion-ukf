import unittest

import numpy as np

from estimator.constants import STATE_DOF
from estimator.data.datamaker import DataMaker
from estimator.data.trajectoryplanner import RoundTripPlanner
from estimator.velocity_integrator import VelocityIntegrator
from estimator.quaternionukf import QuaternionUkf
from estimator.roll_pitch_calculator import RollPitchCalculator


class QuaternionUkfTest(unittest.TestCase):

    def test_ukf_rmse_against_others(self):

        planner = RoundTripPlanner(acc_magnitude=0.0005, noise_stddev=0.02, drift_stddev=0.002)
        data_source = DataMaker(planner)
        R = np.identity(STATE_DOF)
        R[:3, :3] *= planner.noise_stddev ** 2
        R[3:, 3:] *= np.var(planner.drift)
        Q = np.copy(R)

        ukf = QuaternionUkf(data_source, R, Q)
        ukf.estimate_state()

        integrator = VelocityIntegrator(data_source)
        integrator.estimate_state()

        calculator = RollPitchCalculator(data_source)
        calculator.estimate_state()

        estimators = [calculator, integrator, ukf]

        rmse_list = []
        for estimator in estimators:
            rmse_list.append(np.array(estimator.evaluate_estimation()))

        np.testing.assert_array_less(
            rmse_list[2],
            rmse_list[1],
            "UKF performs worse than just integrating gyro data"
        )
        np.testing.assert_array_less(
            rmse_list[2],
            rmse_list[0],
            "UKF roll, pitch calculation performs worse than direct calculation from accelerometer"
        )

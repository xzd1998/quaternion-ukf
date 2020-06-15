import unittest

import numpy as np

from estimator.data.trainer import Trainer
from estimator.data.trajectoryplanner import RoundTripPlanner
from estimator.data.datamaker import DataMaker


class TrainerTest(unittest.TestCase):

    @staticmethod
    def training_method_test(training_method):
        planner = RoundTripPlanner(acc_magnitude=0.0005, noise_stddev=0, drift_stddev=0)
        data_source = DataMaker(planner)
        trainer = Trainer(
            data_source.rots_vicon,
            data_source.imu_data,
            data_source.ts_imu,
            data_source.ts_vicon
        )
        coefficients, _, _ = training_method(trainer)
        np.testing.assert_array_almost_equal(
            coefficients,
            np.ones(len(coefficients)),
            3,
            "Manufactured data with no noise should have unit coefficients, but got: {}"
            .format(coefficients)
        )

    def test_velocity_training(self):

        def training_method_vel(trainer):
            return trainer.train_vel(plot=False)

        self.training_method_test(training_method_vel)

    def test_accelerometer_training(self):

        def training_method_acc(trainer):
            return trainer.train_acc(plot=False)

        self.training_method_test(training_method_acc)

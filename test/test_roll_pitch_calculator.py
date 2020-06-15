import unittest

import numpy as np

from estimator.data import utilities
from estimator.data.datamaker import DataMaker
from estimator.data.trajectoryplanner import RoundTripPlanner
from estimator.roll_pitch_calculator import RollPitchCalculator


class RollPitchCalculatorTest(unittest.TestCase):

    def test_roll_pitch_calculation_no_noise(self):
        planner = RoundTripPlanner(acc_magnitude=0.0005, noise_stddev=0, drift_stddev=0)
        data_source = DataMaker(planner)
        calculator = RollPitchCalculator(data_source)
        calculator.estimate_state()
        roll_expected, pitch_expected, _ = utilities.rots_to_angles_zyx(data_source.rots_vicon)
        roll_actual, pitch_actual, _ = utilities.rots_to_angles_zyx(calculator.rots)
        np.testing.assert_array_almost_equal(roll_actual, roll_expected)
        np.testing.assert_array_almost_equal(pitch_actual, pitch_expected)

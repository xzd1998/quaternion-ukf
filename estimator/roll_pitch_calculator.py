"""
Estimates roll and pitch based on accelerometer data, which are the only angles
that can be calculated from only the accelerometer.
"""

import numpy as np

from estimator.data import utilities
from estimator.state_estimator import StateEstimator


class RollPitchCalculator(StateEstimator):
    """
    Simplest implementation of an estimator - no integration,
    just directly calculates roll and pitch at each timestep
    """

    def estimate_state(self):
        """
        Uses data provided by the source to estimate the state history of the filter
        After calling this function, the state and rotation history will be defined
        """

        roll, pitch = utilities.accs_to_roll_pitch(self.acc_data)
        yaw = np.zeros(roll.shape[0])
        self.rots = utilities.angles_to_rots_zyx(roll, pitch, yaw)

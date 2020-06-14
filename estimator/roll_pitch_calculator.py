import numpy as np

from estimator.data import utilities
from estimator.state_estimator import StateEstimator


class RollPitchCalculator(StateEstimator):

    N_DIM = 6

    def estimate_state(self):

        roll, pitch = utilities.accs_to_roll_pitch(self.acc_data)
        yaw = np.zeros(roll.shape[0])
        self.rots = utilities.angles_to_rots_zyx(roll, pitch, yaw)

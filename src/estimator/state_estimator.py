"""
Defines interface for classes which estimate the state of an object equipped with an IMU.
This state is meant to capture the orientation of, for example, a drone.
"""

from abc import ABC, abstractmethod
from lazy import lazy
import numpy as np
import matplotlib.pyplot as plt

from estimator.data import utilities
from estimator.data.trainer import Trainer


class StateEstimator(ABC):
    """Parent class for both the UKF and the NoFilter implementation"""

    def __init__(self, source):
        self.source = source
        self.rots = None
        self.state = None

    @abstractmethod
    def estimate_state(self):
        """
        Uses data provided by the source to estimate the state history of the filter
        After calling this function, the state and rotation history will be defined
        """

    @property
    def num_data(self):
        """Length of imu data array"""
        return self.source.ts_imu.shape[-1]

    @property
    def ts_imu(self):
        """Times associated with imu data"""
        return self.source.ts_imu

    @lazy
    def imu_data(self):
        """Accelerometer and gyro data from imu"""
        return np.copy(self.source.imu_data)

    @lazy
    def acc_data(self):
        """Acceleromter data"""
        return np.copy(self.source.imu_data[:3])

    @property
    def vel_data(self):
        """Gyro data"""
        return np.copy(self.source.imu_data[3:])

    @staticmethod
    def _normalize_data(data, mag=1):
        return data / np.linalg.norm(data, axis=0) * mag

    def evaluate_estimation(self, plot=False):
        """ Makes 3 plots for roll, pitch, and yaw comparisons of estimated versus truth data """

        indexer = ~np.any(np.isnan(self.source.rots_vicon), axis=(0, 1))
        rots_est, rots_truth, ts_imu, ts_vicon = Trainer.clip_data(
            np.copy(self.source.rots_vicon[..., indexer]),
            np.copy(self.rots),
            np.copy(self.source.ts_imu),
            np.copy(self.source.ts_vicon[indexer])
        )

        labels = ["Roll", "Pitch", "Yaw"]

        angs_est = utilities.rots_to_angles_zyx(rots_est)
        angs_truth = utilities.rots_to_angles_zyx(rots_truth)

        rmse = []
        for (ang_est, ang_truth) in zip(angs_est, angs_truth):
            diff = np.abs(ang_est - ang_truth)
            diff[diff > np.pi] -= 2 * np.pi
            rmse.append(np.sqrt(np.sum(np.square(diff)) / len(ang_est)))

        if plot:
            for i in range(3):
                plt.figure(i)
                plt.plot(ts_vicon, angs_truth[i])
                plt.plot(ts_imu, angs_est[i])
                plt.xlabel("Time [s]")
                plt.ylabel(labels[i] + " Angle [rad]")
                plt.grid(True)
                plt.legend(["Truth", "UKF"])
            plt.show()

        return rmse

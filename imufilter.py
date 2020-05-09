from abc import ABC, abstractmethod
from lazy import lazy
import numpy as np
import matplotlib.pyplot as plt

from data import utilities


class ImuFilter(ABC):
    """Parent class for both the UKF and the NoFilter implementation"""

    def __init__(self, source):
        self.source = source
        self.rots = None

    @abstractmethod
    def filter_data(self):
        """Applies chosen filtering technique to estimate rotation matrices"""
        pass

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
    def angles(self):
        """Tuple of estimated roll, pitch, and yaw angles"""
        return utilities.rots_to_angles(self.rots)

    @lazy
    def acc_data(self):
        """Acceleromter data"""
        return np.copy(self.source.imu_data[:3])

    @lazy
    def vel_data(self):
        """Gyro data"""
        return np.copy(self.source.imu_data[3:])

    @staticmethod
    def plot_comparison(rots_est, ts_imu, rots_vicon, ts_vicon):
        """ Makes 3 plots for roll, pitch, and yaw comparisons of estimated versus truth data """

        rots_truth_copy = np.copy(rots_vicon)
        ts_vicon = np.copy(ts_vicon.reshape(-1))
        t0 = min(ts_vicon[0], ts_imu[0])
        rots_truth_copy = rots_truth_copy[..., ts_vicon > ts_imu[0]]
        rots_est_copy = rots_est[..., ts_imu > ts_vicon[0]]

        labels = ["Roll", "Pitch", "Yaw"]

        a = utilities.rots_to_angles(rots_est_copy)
        angs = utilities.rots_to_angles(rots_truth_copy)

        for i in range(3):
            plt.figure(i)
            plt.plot(ts_vicon[ts_vicon > ts_imu[0]] - t0, angs[i])
            plt.plot(ts_imu[ts_imu > ts_vicon[0]] - t0, a[i])
            plt.xlabel("Time [s]")
            plt.ylabel(labels[i] + " Angle [rad]")
            plt.grid(True)
            plt.legend(["Truth", "UKF"])
        plt.show()

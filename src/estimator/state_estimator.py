"""State Estimator

Defines interface for classes which estimate the state of an object equipped with an IMU.
This state is meant to capture the orientation of, for example, a drone, and it may have
three or six degrees of freedom depending on the implementation.
"""

from abc import ABC, abstractmethod
from lazy import lazy
import numpy as np
import matplotlib.pyplot as plt

from estimator.data import utilities


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

    def _integrate_vel(self):
        angles = np.zeros((3, self.num_data))
        dts = np.diff(self.ts_imu)
        da_dts = (self.vel_data[:, :-1] + self.vel_data[:, 1:]) * dts / 2
        print(da_dts.shape)
        angles[:, 1:] = np.cumsum(da_dts, axis=1)
        return angles

    @property
    def state_dof(self):
        """Number of degrees of freedom in the state vector, which may be overriden to 3"""
        return 6

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
        return utilities.rots_to_angles_zyx(self.rots)

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

    @staticmethod
    def plot_comparison(rots_est, ts_imu, rots_vicon, ts_vicon):
        """ Makes 3 plots for roll, pitch, and yaw comparisons of estimated versus truth data """

        rots_truth_copy = np.copy(rots_vicon)
        ts_vicon = np.copy(ts_vicon.reshape(-1))
        t_start = min(ts_vicon[0], ts_imu[0])
        rots_truth_copy = rots_truth_copy[..., ts_vicon > ts_imu[0]]
        rots_est_copy = rots_est[..., ts_imu > ts_vicon[0]]

        labels = ["Roll", "Pitch", "Yaw"]

        angs_est = utilities.rots_to_angles_zyx(rots_est_copy)
        angs_truth = utilities.rots_to_angles_zyx(rots_truth_copy)

        for i in range(3):
            plt.figure(i)
            plt.plot(ts_vicon[ts_vicon > ts_imu[0]] - t_start, angs_truth[i])
            plt.plot(ts_imu[ts_imu > ts_vicon[0]] - t_start, angs_est[i])
            plt.xlabel("Time [s]")
            plt.ylabel(labels[i] + " Angle [rad]")
            plt.grid(True)
            plt.legend(["Truth", "UKF"])
        plt.show()

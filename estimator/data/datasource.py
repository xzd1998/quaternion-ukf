"""
Defines parent class for where the estimators will pull their data from to perform state
estimation.
"""

from lazy import lazy
import numpy as np

from estimator.data import utilities


class DataSource:
    """
    Parent class for sources of data either real (from a `DataStore`) or made up
    (from a `DataMaker`)

    :ivar ts_vicon: time vector associated with truth data from vicon
    :ivar rots_vicon: rotation matrices from vicon (rotation of robot frame w.r.t. world frame)
    :ivar ts_imu: time vector associated with imu data
    :ivar acc_data: accelerometer data already normalized to be in units of m/s**2
    :ivar vel_data: gyro data already normalized to be in units of rad/s
    """

    def __init__(self, ts_vicon, rots_vicon, ts_imu, acc_data, vel_data):

        self.ts_vicon = ts_vicon
        self.rots_vicon = rots_vicon
        self.ts_imu = ts_imu
        self.acc_data = acc_data
        self.vel_data = vel_data
        self.imu_data = np.vstack((self.acc_data, self.vel_data))

    @lazy
    def angles_vicon(self):
        """Tuple of estimated roll, pitch, and yaw angles"""
        return utilities.rots_to_angles_zyx(self.rots_vicon)

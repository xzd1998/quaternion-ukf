import numpy as np


class DataSource:
    """
    Parent class for sources of data either real (from a `DataStore`) or made up (from a `DataMaker`)
    """
    def __init__(self, ts_vicon, rots_vicon, ts_imu, acc_data, vel_data):
        """
        All parameters become properties of the class
        :param ts_vicon: time vector associated with truth data from vicon
        :param rots_vicon: rotation matrices from vicon (rotation of robot frame w.r.t. world frame)
        :param ts_imu: time vector associated with imu data
        :param acc_data: accelerometer data already normalized to be in units of m/s**2
        :param vel_data: gyro data already normalized to be in units of rad/s
        """
        self.ts_vicon = ts_vicon
        self.rots_vicon = rots_vicon
        self.ts_imu = ts_imu
        self.acc_data = acc_data
        self.vel_data = vel_data
        self.imu_data = np.vstack((self.acc_data, self.vel_data))

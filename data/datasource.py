import numpy as np


class DataSource:
    """
    Parent class for sources of data either real (from a `DataStore`) or made up (from a `DataMaker`)
    """
    def __init__(self, ts_vicon, rots_vicon, ts_imu, accs_imu, vels_imu):
        """
        All parameters become properties of the class
        :param ts_vicon: time vector associated with truth data from vicon
        :param rots_vicon: rotation matrices from vicon (rotation of robot frame w.r.t. world frame)
        :param ts_imu: time vector associated with imu data
        :param accs_imu: accelerometer data already normalized to be in units of m/s**2
        :param vels_imu: gyro data already normalized to be in units of rad/s
        """
        self.ts_vicon = ts_vicon
        self.rots_vicon = rots_vicon
        self.ts_imu = ts_imu
        self.accs_imu = accs_imu
        self.vels_imu = vels_imu
        self.data_imu = np.vstack((self.accs_imu, self.vels_imu))

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat
from scipy import constants

from data.datamaker import DataMaker
from data import utilities


class DataStore:

    def __init__(self, dataset_number, path_to_data="."):
        self.dataset_number = dataset_number
        self.path_to_data = path_to_data

        self.imu_filename = os.path.join(path_to_data, "imu", "imuRaw{}.mat".format(self.dataset_number))
        self.vicon_filename = os.path.join(path_to_data, "vicon", "viconRot{}.mat".format(self.dataset_number))

        imu_data = loadmat(self.imu_filename)
        vicon_data = loadmat(self.vicon_filename)
        sensor_data = imu_data["vals"].astype(float)

        self.t_vicon = vicon_data["ts"].reshape(-1)
        self.t_imu = imu_data["ts"].reshape(-1)

        self.rots = vicon_data["rots"]
        self.vals = np.copy(sensor_data)

        # reorder gyro data from imu to roll-pitch-yaw convention
        self.vals[3] = sensor_data[4]
        self.vals[4] = sensor_data[5]
        self.vals[5] = sensor_data[3]

    @property
    def angs(self):
        return utilities.rots_to_angles(self.rots)

    @property
    def zs_vicon(self):
        return self.rots[:, -1]

    @property
    def gs_vicon(self):
        g = constants.g
        return self.zs_vicon * np.array([-g, -g, g]).reshape(3, 1)

    def gs_imu(self, m, b):
        return self.vals[:3] * m + b

    @property
    def vels(self):
        return self.vals[3:]


if __name__ == "__main__":
    mr = np.array([-0.09363796, -0.09438229, 0.09449341]).reshape(3, 1)
    br = np.array([47.88161084, 47.23512485, -47.39899347]).reshape(3, 1)
    store = DataStore(1)
    utilities.plot_rowwise_data(
        ["Truth", "Measured"],
        ["x", "y", "z"],
        [store.t_vicon, store.t_imu],
        store.gs_vicon,
        store.gs_imu(mr, br)
    )

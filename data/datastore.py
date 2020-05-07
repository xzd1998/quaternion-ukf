import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat
from scipy import constants

from data.datamaker import DataMaker
from data import utilities
from data.datasource import DataSource
from data.trainer import Trainer


class DataStore(DataSource):
    """
    Loads data from an existing source of matching vicon and imu data
    """
    def __init__(self, dataset_number, path_to_data=".", m=Trainer.m, b=Trainer.b):
        """
        :param m: linear coefficients for imu data
        :param b: biases for imu data
        :param dataset_number: corresponding to the numbers in the filenames
        :param path_to_data: path to the directory of this package
        """
        if m.shape != (6,):
            raise ValueError(
                "Value coefficients have invalid shape: expected {} but got {}".format((6,), m.shape)
            )
        if b.shape != (6,):
            raise ValueError(
                "Value biases have invalid shape: expected {} but got {}".format((6,), b.shape)
            )

        self.dataset_number = dataset_number
        self.path_to_data = path_to_data

        self.imu_filename = os.path.join(path_to_data, "imu", "imuRaw{}.mat".format(self.dataset_number))
        self.vicon_filename = os.path.join(path_to_data, "vicon", "viconRot{}.mat".format(self.dataset_number))

        imu_data = loadmat(self.imu_filename)
        vicon_data = loadmat(self.vicon_filename)
        sensor_data = imu_data["vals"].astype(float)

        ts_vicon = vicon_data["ts"].reshape(-1)
        ts_imu = imu_data["ts"].reshape(-1)

        rots_vicon = vicon_data["rots"]
        data_imu = np.copy(sensor_data)

        # reorder gyro data from imu to roll-pitch-yaw convention
        data_imu[3] = sensor_data[4]
        data_imu[4] = sensor_data[5]
        data_imu[5] = sensor_data[3]
        data_imu = data_imu * m.reshape(-1, 1) + b.reshape(-1, 1)

        super().__init__(ts_vicon, rots_vicon, ts_imu, data_imu[:3], data_imu[3:])


if __name__ == "__main__":
    from data.trainer import Trainer, Trainer

    store = DataStore(dataset_number=1)

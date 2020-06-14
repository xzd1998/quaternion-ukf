"""
In contrast to :code:`DataMaker`s, :code:`DataStore`s load real sensor data.
"""

import os

import numpy as np
from scipy.io import loadmat

from estimator.data.datasource import DataSource
from estimator.data.trainer import Trainer
from estimator.constants import STATE_DOF


class DataStore(DataSource):  # pylint: disable=too-few-public-methods
    """
    Loads data from an existing source of matching vicon and imu data. If the intention is
    to calibrate the data, :code:`coefficients` and :code:`intercepts` should be specified
    as :code:`None`, otherwise the :code:`DataSource` will automatically calibrate the data
    using the static members defined in :doc:`./trainer`

    :ivar dataset_number: corresponding to the numbers in the filenames
    :ivar path_to_data: path to the directory of this package
    :ivar coefficients: linear coefficients for imu data
                         may be set to none to leave data uncalibrated
    :ivar intercepts: biases for imu data
                       may be set to none to leave data uncalibrated
    """
    def __init__(
            self,
            dataset_number,
            path_to_data=".",
            coefficients=Trainer.IMU_COEFFICIENTS,
            intercepts=Trainer.IMU_INTERCEPTS
    ):
        if coefficients is not None and coefficients.shape != (STATE_DOF,):
            raise ValueError(
                "Value coefficients are invalid: expected {} but got {}"
                .format((STATE_DOF,), coefficients.shape)
            )
        if intercepts is not None and intercepts.shape != (STATE_DOF,):
            raise ValueError(
                "Value biases are invalid: expected {} but got {}"
                .format((STATE_DOF,), intercepts.shape)
            )

        self.dataset_number = dataset_number
        self.path_to_data = path_to_data

        self.imu_filename = os.path.join(
            path_to_data,
            "imu",
            "imuRaw{}.mat".format(self.dataset_number)
        )
        self.vicon_filename = os.path.join(
            path_to_data,
            "vicon",
            "viconRot{}.mat".format(self.dataset_number)
        )

        imu_data = loadmat(self.imu_filename)
        vicon_data = loadmat(self.vicon_filename)
        sensor_data = imu_data["vals"].astype(float)

        ts_vicon = vicon_data["ts"].reshape(-1)
        ts_imu = imu_data["ts"].reshape(-1)

        rots_vicon = vicon_data["rots"]
        imu_data = np.copy(sensor_data)

        # reorder gyro data from imu to roll-pitch-yaw convention
        imu_data[3] = sensor_data[4]
        imu_data[4] = sensor_data[5]
        imu_data[5] = sensor_data[3]
        if coefficients is not None and intercepts is not None:
            imu_data = imu_data * coefficients.reshape(-1, 1) + intercepts.reshape(-1, 1)
            imu_data[3:] -= imu_data[3:, :1]

        super().__init__(ts_vicon, rots_vicon, ts_imu, imu_data[:3], imu_data[3:])

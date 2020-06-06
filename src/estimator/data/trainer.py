"""
Trainer

Calibrates IMU data using linear regression after transforming ground-truth data,
which comes in the form of rotation matrices. The rotation matrix data is
transformed into the expected gravity vector and expected angular velocity vector
to be able to calibrate the accelerometer and gyroscope, respectively.

The trainer has two static members that are the results of training on the first
three datasets:
* `IMU_COEFFICIENTS`: ith coeffient multiplies the ith row of `imu_data`
* `IMU_INTERCEPTS`: ith intercept is added to the ith row of `imu_data`

The convention above is slightly different from that listed in the IMU reference:
:doc:`IMU reference <../../../docs/IMU_reference.pdf>`

.. code-block::
   :linenos:

   from estimator.data.datastore import DataStore
   # Note: DataStores calibrates data unless coeffs and inters specified as None
   store = DataStore(dataset_number=3, coefficients=None, intercepts=None)
   trainer = Trainer(
       store.rots_vicon,
       store.imu_data,
       store.ts_imu,
       store.ts_vicon
   )
   coeffs, inters, coef_det = trainer.train_vel()
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.constants

from estimator.data import utilities
from estimator.constants import NUM_AXES


class Trainer:
    """Interface for training a linear regression model for accelerometer and gyro data"""

    # Results from combined training on first three datasets
    IMU_COEFFICIENTS = np.array([-0.0936, -0.0944, 0.0945, 0.0155, 0.0158, 0.0161])
    IMU_INTERCEPTS = np.array([47.9, 47.2, -47.4, -5.75, -5.75, -5.95])

    def __init__(self, rots, imu_data, ts_imu, ts_vicon):
        """
        :param rots: (3, 3, N) rotation matrices w.r.t. world frame, treated as the ground truth
        :param imu_data: (6, N) raw data, three rows for accelerometer and then three for gyro
        :param ts_imu: (N,) time vector associated with the raw data
        :param ts_vicon: (N,) time vector associated with the truth data
        """

        indexer = np.logical_not(
            np.array([np.any(np.isnan(rots[..., i])) for i in range(rots.shape[-1])])
        )
        rots = rots[..., indexer]
        ts_vicon = ts_vicon[indexer]

        self.rots, self.imu_data, self.t_imu, self.t_vicon = \
            Trainer.clip_data(rots, imu_data, ts_imu, ts_vicon)

    def train_acc(self):
        """Solves for coefficients for accelerometer data"""

        coefficients = np.zeros(NUM_AXES)
        intercepts = np.zeros(NUM_AXES)

        # Perform a least squares estimate of the accelerometer parameters
        # First, get expected acceleration vectors from rotation matrices and gravity field
        g_vector = np.array([0, 0, scipy.constants.g]).reshape(3, 1)
        truth = np.zeros((NUM_AXES, 1, self.rots.shape[-1]))
        for i in range(self.rots.shape[-1]):
            truth[..., i] = np.matmul(self.rots[..., i].T, g_vector)

        for i in range(NUM_AXES):
            training_data = np.vstack([self.imu_data[i], np.ones(self.rots.shape[-1])]).T
            coefficients[i], intercepts[i] = np.linalg.lstsq(
                training_data,
                truth[i].reshape(-1, 1),
                rcond=None
            )[0]

        measured = np.array([
            [coefficients[i] * self.imu_data[i] + intercepts[i]] for i in range(NUM_AXES)
        ])

        for i in range(NUM_AXES):
            plt.scatter(measured[i].reshape(-1), truth[i].reshape(-1))
            plt.show()

        coef_determination = np.identity(NUM_AXES) * np.var(measured - truth, axis=2)

        for i in range(3):
            plt.plot(self.t_vicon.reshape(-1), measured[i].reshape(-1))
            plt.plot(self.t_vicon.reshape(-1), truth[i].reshape(-1))
            plt.show()

        return coefficients, intercepts, coef_determination

    def train_vel(self):
        """Solves for coefficients for gyro data"""

        coefficients = np.zeros(NUM_AXES)
        intercepts = np.zeros(NUM_AXES)
        vels, t_vicon = utilities.rots_to_vels(self.rots, self.t_vicon)
        vels = utilities.moving_average(vels, 9)
        vels, imu_data, _, t_vicon = Trainer.clip_data(vels, self.imu_data, self.t_imu, t_vicon)

        for i in range(3):
            training_data = np.vstack(
                [imu_data[i + NUM_AXES], np.ones(imu_data[i + NUM_AXES].shape[-1])]
            ).T
            coefficients[i], intercepts[i] = np.linalg.lstsq(training_data, vels[i], rcond=None)[0]

        coefficients[-1] = coefficients[1]
        intercepts[-1] = intercepts[1]
        measured = np.array([
            [coefficients[i] * imu_data[i + NUM_AXES] + intercepts[i]] for i in range(3)
        ])
        coef_determination = np.identity(3) * np.var(measured - vels, axis=2)

        measured = measured.reshape(NUM_AXES, -1)
        for i in range(NUM_AXES):
            plt.plot(t_vicon.reshape(-1), measured[i].reshape(-1))
            plt.plot(t_vicon.reshape(-1), vels[i].reshape(-1))
            plt.legend(["Measured", "Truth"])
            plt.show()

        return coefficients, intercepts, coef_determination

    @staticmethod
    def clip_data(data_to_clip, reference_data, ts_imu, ts_vicon):
        """
        Lines up both time vectors and interpolates the data to clip to the raw data
        Because there's no guarantee that the two time vectors are the same
        """

        if ts_imu[0] < ts_vicon[0]:
            ts_vicon -= ts_imu[0]
            ts_imu -= ts_imu[0]
            indexer = ts_imu >= ts_vicon[0]
            ts_imu = ts_imu[indexer]
            reference_data = reference_data[..., indexer]

        if ts_imu[-1] > ts_vicon[-1]:
            indexer = ts_imu <= ts_vicon[-1]
            ts_imu = ts_imu[indexer]
            reference_data = reference_data[..., indexer]

        interp_func = interp1d(ts_vicon, data_to_clip)
        clipped = interp_func(ts_imu)
        ts_vicon = ts_imu

        return clipped, reference_data, ts_imu, ts_vicon


if __name__ == "__main__":
    from estimator.data.datastore import DataStore

    store = DataStore(dataset_number=3, coefficients=None, intercepts=None)

    trainer = Trainer(store.rots_vicon, store.imu_data, store.ts_imu, store.ts_vicon)

    coeffs, inters, coef_det = trainer.train_vel()
    print(coeffs)
    print(inters)
    print(coef_det)

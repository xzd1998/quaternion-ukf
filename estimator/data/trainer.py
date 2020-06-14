"""
Calibrates IMU data using linear regression after transforming ground-truth data,
which comes in the form of rotation matrices. The rotation matrix data is
transformed into the expected gravity vector and expected angular velocity vector
to be able to calibrate the accelerometer and gyroscope, respectively.

The trainer has two static members that are the results of training on the first
three datasets:

* :code:`IMU_COEFFICIENTS`: ith coeffient multiplies the ith row of `imu_data`
* :code:`IMU_INTERCEPTS`: ith intercept is added to the ith row of `imu_data`

The convention above is slightly different from that listed in the IMU reference
`here <https://github.com/mattlisle/quaternion-ukf/blob/master/docs/IMU_reference.pdf>`_,
in that the formula they list is :code:`value = (raw - bias) * scale_factor` whereas
the parameters learned in :code:`Trainer` correspond to this formula:
:code:`value = raw * scale_factor + bias`

The example below uses the 3rd dataset (imuRaw3.mat and viconRot3.mat) to calculate
the calibrated velocity data from the IMU. Note that when :code:`DataStore` is used
in practice, :code:`Trainer`'s :code:`IMU_COEFFICIENTS` and :code:`IMU_INTERCEPTS`
are used to automatically calibrate the data. That way, :code:`store.vel_data` has
already been calibrated.

.. code-block::
   :linenos:

   from estimator.data.datastore import DataStore
   # Note: DataStore calibrates data by default unless coeffs and inters specified as None
   store = DataStore(dataset_number=3, coefficients=None, intercepts=None)
   trainer = Trainer(
       store.rots_vicon,
       store.imu_data,
       store.ts_imu,
       store.ts_vicon
   )
   coeffs, biases, coef_determination = trainer.train_vel()
   calibrated_vel_data = store.vel_data * coeffs.reshape(-1, 1) + biases.reshape(-1, 1)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.constants

from estimator.data import utilities
from estimator.constants import NUM_AXES


class Trainer:
    """
    Interface for training a linear regression model for accelerometer and gyro data
    with two static members defined such that:

    .. code-block::

       calibrated = raw * IMU_COEFFICIENTS.reshape(-1, 1) + IMU_INTERCEPTS.reshape(-1, 1)

    :cvar IMU_COEFFICIENTS: ith coeffient multiplies the ith row of `imu_data`
    :cvar IMU_INTERCEPTS: ith coeffient is added to the ith row of `imu_data`
    """
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
        """
        Solves for coefficients for accelerometer data

        :return: accelerometer coefficients, biases, coefficients of determination
        """

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

        coef_determination = np.var(measured - truth, axis=2)

        for i in range(3):
            plt.plot(self.t_vicon.reshape(-1), measured[i].reshape(-1))
            plt.plot(self.t_vicon.reshape(-1), truth[i].reshape(-1))
            plt.show()

        return coefficients, intercepts, coef_determination

    def train_vel(self):
        """
        Solves for coefficients for accelerometer data

        :return: gyro coefficients, biases, coefficients of determination
        """

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
        coef_determination = np.var(measured - vels, axis=2)

        measured = measured.reshape(NUM_AXES, -1)
        for i in range(NUM_AXES):
            plt.plot(t_vicon.reshape(-1), measured[i].reshape(-1))
            plt.plot(t_vicon.reshape(-1), vels[i].reshape(-1))
            plt.legend(["Measured", "Truth"])
            plt.show()

        return coefficients, intercepts, coef_determination

    @staticmethod
    def clip_data(data_to_clip, data_reference, ts_reference, ts_to_clip):
        """
        Lines up both time vectors and interpolates the data to clip to the raw data
        because there's no guarantee that the two time vectors are the same

        :param data_to_clip: for example, vicon data
        :param data_reference: data to which the data to clip will be interpolated
        :param ts_reference: time vector associated with reference data
        :param ts_to_clip: time vector associated with data to clip
        :return: all of the above parameters after clipping/interpolation
        """

        if ts_reference[0] < ts_to_clip[0]:
            ts_to_clip -= ts_reference[0]
            ts_reference -= ts_reference[0]
            indexer = ts_reference >= ts_to_clip[0]
            ts_reference = ts_reference[indexer]
            data_reference = data_reference[..., indexer]

        if ts_reference[-1] > ts_to_clip[-1]:
            indexer = ts_reference <= ts_to_clip[-1]
            ts_reference = ts_reference[indexer]
            data_reference = data_reference[..., indexer]

        interp_func = interp1d(ts_to_clip, data_to_clip)
        clipped = interp_func(ts_reference)
        ts_to_clip = ts_reference

        return clipped, data_reference, ts_reference, ts_to_clip

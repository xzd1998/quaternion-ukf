import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from data import utilities
from data.trajectoryplanner import SimplePlanner


class Trainer:
    """
    Interface for training a linear regression model for accelerometer and gyro data
    """

    # Results from combined training on first three datasets
    m = np.array([-0.0936, -0.0944, 0.0945, 0.0155, 0.0158, 0.0161])
    b = np.array([47.9, 47.2, -47.4, -5.75, -5.75, -5.95])

    def __init__(self, rots, vals, ts_imu, ts_vicon):
        """
        :param rots: (3, 3, N) rotation matries with respect to the world frame to be treated as the ground truth
        :param vals: (6, N) raw data, first three rows are accelerometer data and others the gyro data
        :param ts_imu: (N,) time vector associated with the raw data
        :param ts_vicon: (N,) time vector associated with the truth data
        """

        indexer = np.logical_not(np.array([np.any(np.isnan(rots[..., i])) for i in range(rots.shape[-1])]))
        rots = rots[..., indexer]
        ts_vicon = ts_vicon[indexer]

        self.rots, self.vals, self.t_imu, self.t_vicon = Trainer.clip_data(rots, vals, ts_imu, ts_vicon)

    def train_acc(self):
        """Solves for coefficients for accelerometer data"""

        m = np.zeros(3)
        b = np.zeros(3)

        # Perform a least squares estimate of the accelerometer parameters
        # First, get expected acceleration vectors from rotation matrices and gravity field
        g = np.array([0, 0, 9.81]).reshape(3, 1)
        truth = np.zeros((3, 1, self.rots.shape[2]))
        for i in range(self.rots.shape[2]):
            truth[..., i] = np.matmul(self.rots[..., i].T, g)

        for i in range(3):
            A = np.vstack([self.vals[i], np.ones(self.rots.shape[2])]).T
            m[i], b[i] = np.linalg.lstsq(A, truth[i].reshape(-1, 1), rcond=None)[0]

        measured = np.array([[m[i] * self.vals[i] + b[i]] for i in range(3)])

        for i in range(3):
            plt.scatter(measured[i].reshape(-1), truth[i].reshape(-1))
            plt.show()

        R = np.identity(3) * np.var(measured - truth, axis=2)

        for i in range(3):
            plt.plot(self.t_vicon.reshape(-1), measured[i].reshape(-1))
            plt.plot(self.t_vicon.reshape(-1), truth[i].reshape(-1))
            plt.show()

        return m, b, R

    def train_vel(self):
        """Solves for coefficients for gyro data"""

        m = np.zeros(3)
        b = np.zeros(3)
        vels, t_vicon = utilities.rots_to_vels(self.rots, self.t_vicon)
        vels = utilities.moving_average(vels, 9)
        vels, vals, t_imu, t_vicon = Trainer.clip_data(vels, self.vals, self.t_imu, t_vicon)

        for i in range(3):
            A = np.vstack([vals[i + 3], np.ones(vals[i + 3].shape[-1])]).T
            m[i], b[i] = np.linalg.lstsq(A, vels[i], rcond=None)[0]

        m[-1] = m[1]
        b[-1] = b[1]
        measured = np.array([[m[i] * vals[i + 3] + b[i]] for i in range(3)])
        R = np.identity(3) * np.var(measured - vels, axis=2)

        measured = measured.reshape(3, -1)
        for i in range(3):
            plt.plot(t_vicon.reshape(-1), measured[i].reshape(-1))
            plt.plot(t_vicon.reshape(-1), vels[i].reshape(-1))
            plt.legend(["Measured", "Truth"])
            plt.show()

        return m, b, R

    @staticmethod
    def clip_data(to_clip, vals, ts_imu, ts_vicon):
        """
        Lines up both time vectors and interpolates the data to clip to the raw data
        Because there's no guarantee that the two time vectors are the same
        """

        if ts_imu[0] < ts_vicon[0]:
            ts_vicon -= ts_imu[0]
            ts_imu -= ts_imu[0]
            indexer = ts_imu >= ts_vicon[0]
            ts_imu = ts_imu[indexer]
            vals = vals[..., indexer]

        if ts_imu[-1] > ts_vicon[-1]:
            indexer = ts_imu <= ts_vicon[-1]
            ts_imu = ts_imu[indexer]
            vals = vals[..., indexer]

        interp_func = interp1d(ts_vicon, to_clip)
        clipped = interp_func(ts_imu)
        ts_vicon = ts_imu

        return clipped, vals, ts_imu, ts_vicon


if __name__ == "__main__":
    from data.datastore import DataStore
    from data.datamaker import DataMaker

    store = DataStore(dataset_number=1)
    planner = SimplePlanner()
    maker = DataMaker(planner)

    trainer = Trainer(store.rots_vicon, store.imu_data, store.ts_imu, store.ts_vicon)
    # trainer = Trainer(maker.rots_g, maker.vals, maker.ts_imu, maker.ts_vicon)

    m, b, R = trainer.train_vel()
    print(m)
    print(b)
    print(R)

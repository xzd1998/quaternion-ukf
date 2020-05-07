import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from data import utilities
from data.trajectoryplanner import SimplePlanner


class Trainer:
    """
    Interface for training a linear regression model for accelerometer and gyro data
    """

    m = np.array([-0.09363796, -0.09438229, 0.09449341, 0.01546466, 0.01578361, 0.01610787])
    b = np.array([47.88161084, 47.23512485, -47.39899347, -5.75, -5.75, -5.9])

    def __init__(self, rots, vals, t_imu, t_vicon):
        """
        :param rots: (3, 3, N) rotation matries with respect to the world frame to be treated as the ground truth
        :param vals: (6, N) raw data, first three rows are accelerometer data and others the gyro data
        :param t_imu: (N,) time vector associated with the raw data
        :param t_vicon: (N,) time vector associated with the truth data
        """

        indexer = np.logical_not(np.array([np.any(np.isnan(rots[..., i])) for i in range(rots.shape[-1])]))
        rots = rots[..., indexer]
        t_vicon = t_vicon[indexer]

        self.rots, self.vals, self.t_imu, self.t_vicon = Trainer.clip_data(rots, vals, t_imu, t_vicon)

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
        vels_g, t_vicon = utilities.rots_to_vels(self.rots, self.t_vicon)
        vels_g = utilities.moving_average(vels_g, 9)
        vels_g, vals, t_imu, t_vicon = Trainer.clip_data(vels_g, self.vals, self.t_imu, t_vicon)

        for i in range(3):
            A = np.vstack([vals[i + 3], np.ones(vals[i + 3].shape[-1])]).T
            m[i], b[i] = np.linalg.lstsq(A, vels_g[i], rcond=None)[0]

        m[-1] = m[1]
        b[-1] = b[1]
        measured = np.array([[m[i] * vals[i + 3] + b[i]] for i in range(3)])
        R = np.identity(3) * np.var(measured - vels_g, axis=2)

        measured = measured.reshape(3, -1)
        for i in range(3):
            plt.plot(t_vicon.reshape(-1), measured[i].reshape(-1))
            plt.plot(t_vicon.reshape(-1), vels_g[i].reshape(-1))
            plt.legend(["Measured", "Truth"])
            plt.show()

        return m, b, R

    @staticmethod
    def clip_data(to_clip, vals, t_imu, t_vicon):
        """
        Lines up both time vectors and interpolates the data to clip to the raw data
        Because there's no guarantee that the two time vectors are the same
        """

        if t_imu[0] < t_vicon[0]:
            t_vicon -= t_imu[0]
            t_imu -= t_imu[0]
            indexer = t_imu >= t_vicon[0]
            t_imu = t_imu[indexer]
            vals = vals[..., indexer]

        if t_imu[-1] > t_vicon[-1]:
            indexer = t_imu <= t_vicon[-1]
            t_imu = t_imu[indexer]
            vals = vals[..., indexer]

        interp_func = interp1d(t_vicon, to_clip)
        clipped = interp_func(t_imu)
        t_vicon = t_imu

        return clipped, vals, t_imu, t_vicon


if __name__ == "__main__":
    from data.datastore import DataStore
    from data.datamaker import DataMaker

    store = DataStore(dataset_number=1)
    planner = SimplePlanner()
    maker = DataMaker(planner)

    trainer = Trainer(store.rots_vicon, store.data_imu, store.ts_imu, store.ts_vicon)
    # trainer = Trainer(maker.rots_g, maker.vals, maker.ts_imu, maker.ts_vicon)

    m, b, R = trainer.train_vel()
    print(m)
    print(b)
    print(R)

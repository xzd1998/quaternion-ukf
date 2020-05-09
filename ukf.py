import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from data import utilities
from data.datamaker import DataMaker
from data.datastore import DataStore
from data.trajectoryplanner import SimplePlanner, StationaryPlanner
from imufilter import ImuFilter
from quaternions import Quaternions


class Ukf(ImuFilter):

    N_DIM = 6

    def __init__(self, source, R, Q):

        super().__init__(source)

        self.R = R
        self.Q = Q

        # Initialize covariance history and state history
        self.mu = np.zeros((Ukf.N_DIM + 1, self.imu_data.shape[-1]))
        self.mu[:, 0] = np.array([1, 0, 0, 0, 0, 0, 0])
        self.P = np.zeros((Ukf.N_DIM, Ukf.N_DIM, self.mu.shape[-1]))
        self.P[..., 0] = np.identity(Ukf.N_DIM) * .5

    def filter_data(self):

        self.imu_data[3:, :] = self.imu_data[3:, :] - ((np.mean(self.imu_data[3:, :50], axis=1) +
                                                        np.mean(self.imu_data[3:, -50:], axis=1)) / 2).reshape(3, 1)
        # TODO this seems like a problem
        self.imu_data[:3] = self.imu_data[:3] / np.linalg.norm(self.imu_data[:3], axis=0)

        bot_rots = np.zeros((3, 3, self.mu.shape[-1]))
        bot_rots[..., 0] = Quaternions(self.mu[:4, 0]).to_rotation_matrix()

        for i in range(1, self.mu.shape[-1]):
            dt = self.ts_imu[i] - self.ts_imu[i - 1]

            self.mu[:, i], self.P[..., i] = self._filter_next(
                self.P[..., i - 1],
                self.mu[:, i - 1],
                self.imu_data[:, i],
                dt
            )
            bot_rots[..., i] = Quaternions(self.mu[:4, i]).to_rotation_matrix()

        self.rots = bot_rots

    def _filter_next(self, P_last, mu_last, z_this, dt):

        # 2n sigma points
        n = P_last.shape[0]
        m = np.sqrt(n)
        S = np.linalg.cholesky(P_last + self.Q)
        W = np.concatenate((m * S, -m * S), axis=1)
        W = np.concatenate((np.zeros((n, 1)), W), axis=1)

        # Equation 34: Form sigma points based on prior mean and covariance data
        qW = Quaternions.from_vector(W[:3])
        q_last = Quaternions(mu_last[:4])
        q_sigpt = q_last.q_multiply(qW)

        wW = W[3:]
        w_last = mu_last[4:]
        w_sigpt = w_last.reshape(-1, 1) * np.ones(wW.shape) + wW

        # Equations 9-11: form q_delta
        ad = np.linalg.norm(w_sigpt, axis=0) * dt
        ed = np.zeros(w_sigpt.shape)
        if np.any(ad == 0):
            ind = ad == 0
            not_ind = ad != 0
            ed[:, ind] = np.zeros((3, np.sum(ind)))
            ed[:, not_ind] = w_sigpt[:, not_ind].astype(float) * dt / ad[not_ind]
        else:
            ed = -w_sigpt.astype(float) * dt / ad
        qd = np.array([np.cos(ad * .5), ed[0] * np.sin(ad * .5), ed[1] * np.sin(ad * .5), ed[2] * np.sin(ad * .5)])
        qd = Quaternions(qd.astype(float) / np.linalg.norm(qd, axis=0))

        # Equation 22: Apply non-linear function A with process noise of zero
        qY = q_sigpt.q_multiply(qd)
        Y = np.concatenate((qY.array, w_sigpt))

        # Equations 52-55: Use mean-finding algorithm to satisfy Equation 38
        Y[:4, :] = Y[:4, :] * np.sign(Y[0, :])
        q1 = Quaternions(Y[:4, 0])
        qs = Quaternions(Y[:4, :])
        q_mean = qs.find_q_mean(q1)
        w_mean = np.mean(Y[4:, :], axis=1)
        mu_this_est = np.concatenate((q_mean.array.reshape(-1), w_mean.reshape(-1)))

        # Equations 65-67: Transform Y into W', notated as Wp for prime
        rWp = q_mean.inverse().q_multiply(qs).to_vectors()
        # rWp = qs.q_multiply(q_mean.inverse()).to_vectors()
        wWp = Y[4:, :] - w_mean.reshape(-1, 1)
        Wp = np.concatenate((rWp, wWp))

        # Equation 64
        Pk_bar = np.matmul(Wp, Wp.T)

        # Equation 27 and 40
        g = Quaternions(np.array([0, 0, 0, 1]))
        gp = qs.inverse().q_multiply(g).q_multiply(qs)
        Z = np.concatenate((gp.array[1:], Y[4:, :]))

        Pk_bar /= W.shape[1]

        # Equation 48
        z_est = np.mean(Z, axis=1)

        # Equation 68
        # Equation 70
        z_diff = Z - z_est.reshape(-1, 1)
        Pzz = np.matmul(z_diff, z_diff.T)
        Pxz = np.matmul(Wp, z_diff.T)
        Pzz /= W.shape[1]
        Pxz /= W.shape[1]

        # Equation 69
        Pvv = Pzz + self.R

        # Equation 72
        K = np.matmul(Pxz, np.linalg.inv(Pvv))

        # Equation 74
        r_this = np.matmul(K, (z_this - z_est).reshape(-1, 1)).reshape(-1)
        q_this = Quaternions.from_vector(r_this[:3])
        w_this = r_this[3:]

        # Equation 46
        mu_this = np.zeros(mu_this_est.shape)
        mu_this[:4] = Quaternions(mu_this_est[:4]).q_multiply(q_this).array
        mu_this[4:] = mu_this_est[4:] + w_this

        # Equation 75:
        P_this = Pk_bar - np.matmul(np.matmul(K, Pvv), K.T)

        return mu_this, P_this


if __name__ == "__main__":
    from data.trainer import Trainer
    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--datanum", required=False, help="Number of data file (1 to 3 inclusive)")

    args = vars(parser.parse_args())

    # Noise parameters for UKF
    Rr = np.array([.05, .05, .15])
    Rw = np.array([.01 for i in range(3)])
    R = np.identity(Ukf.N_DIM) * np.concatenate((Rr, Rw))
    Q = np.identity(Ukf.N_DIM) * 2.993
    Q[:3, :3] *= 2

    num = args["datanum"]
    if not num:
        planner = SimplePlanner()
        source = DataMaker(planner)
    else:
        source = DataStore(dataset_number=num, path_to_data="data")

    f = Ukf(source, R, Q)
    f.filter_data()

    if not num:
        utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.angs_vicon, f.angles)
    else:
        ImuFilter.plot_comparison(f.rots, f.ts_imu, source.rots_vicon, source.ts_vicon)
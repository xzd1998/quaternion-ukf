import argparse

import numpy as np
import matplotlib.pyplot as plt

from data import utilities
from data.datamaker import DataMaker
from data.datastore import DataStore
from data.trajectoryplanner import SimplePlanner, StationaryPlanner, RoundTripPlanner
from imufilter import ImuFilter
from quaternions import Quaternions


class QuaternionUkf(ImuFilter):

    n = 6
    g_vector = np.array([0, 0, 1])

    def __init__(self, source, R, Q, alpha=1, beta=2, kappa=2):

        super().__init__(source)

        self.R = R
        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Initialize covariance history and state history
        self.mu = np.zeros((self.n + 1, self.imu_data.shape[-1]))
        self.mu[:, 0] = np.array([1, 0, 0, 0, 0, 0, 0])
        self.P = np.zeros((self.n, self.n, self.mu.shape[-1]))
        self.P[..., 0] = np.identity(self.n) * .5

        self._t = 0

        self.free_param = self.alpha ** 2 * (self.n + self.kappa) - self.n

        w0m = self.free_param / (self.n + self.free_param)
        qs_to_pad = 2 * self.n * w0m / (1 - w0m) - 1
        if qs_to_pad - np.round(qs_to_pad) < 1e-5:
            self.qs_to_pad = int(np.round(qs_to_pad))
        else:
            raise ValueError("Can't weight with fraction of quaternion to find quaternion mean")

        w0c = w0m + 1 - self.alpha ** 2 + self.beta
        wi = 1 / (2 * (self.n + self.free_param))

        self.weights_mu = np.insert(np.ones(2 * self.n) * wi, 0, w0m)
        self.weights_p = np.insert(np.ones(2 * self.n) * wi, 0, w0c)

    def _debug_print(self, t_min, t_max, *contents):
        if t_min <= self._t <= t_max:
            print("Time {} seconds".format(self._t))
            for content in contents:
                print(content)

    def _get_sigma_distances(self, P_last):
        m = self.n + self.free_param
        S = np.linalg.cholesky(m * P_last)
        W = np.concatenate((S, -S), axis=1)
        return np.concatenate((np.zeros((self.n, 1)), W), axis=1)

    def _get_custom_distances(self):
        S = np.identity(self.n) * 3
        S[:3] /= 30
        W = np.concatenate((S, -S), axis=1)
        return np.concatenate((np.zeros((self.n, 1)), W), axis=1)

    def filter_data(self):

        self.imu_data[:3] = self._normalize_data(self.imu_data[:3])

        rots = np.zeros((3, 3, self.mu.shape[-1]))
        rots[..., 0] = Quaternions(self.mu[:4, 0]).to_rotation_matrix()

        for i in range(1, self.mu.shape[-1]):
            dt = self.ts_imu[i] - self.ts_imu[i - 1]
            self._t = self.ts_imu[i]

            self.mu[:, i], self.P[..., i] = self._filter_next(
                self.P[..., i - 1],
                self.mu[:, i - 1],
                self.imu_data[:, i],
                dt
            )
            rots[..., i] = Quaternions(self.mu[:4, i]).to_rotation_matrix()

        self.rots = rots

    def _filter_next(self, P_last, mu_last, z_this, dt):

        # W = self._get_sigma_distances(P_last)
        W = self._get_custom_distances()
        # self._debug_print(11.7, 12, np.round(P_last, 2), np.round(S, 2))

        # Equation 34: Form sigma points based on prior mean and covariance data
        qW = Quaternions.from_vectors(W[:3])
        q_last = Quaternions(mu_last[:4])
        q_sigpt = qW.q_multiply(q_last)

        wW = W[3:]
        w_last = mu_last[4:]
        w_sigpt = w_last.reshape(-1, 1) + wW

        # Equations 9-11: form q_delta
        qd = Quaternions.from_vectors(w_sigpt * dt)

        # Equation 22: Apply non-linear function A with process noise of zero
        qY = qd.q_multiply(q_sigpt)
        Y = np.concatenate((qY.array, w_sigpt))

        # Equations 52-55: Use mean-finding algorithm to satisfy Equation 38
        Y[:4] = Y[:4] * np.sign(Y[0])
        q1 = Quaternions(Y[:4, 0])
        qs = Quaternions(Y[:4])
        q_mean = qs.find_q_mean(q1)

        # extra_q1s = np.matmul(q1.array.reshape(-1, 1), np.ones((1, self.qs_to_pad)))
        # qs_padded = Quaternions(np.concatenate((extra_q1s, Y[:4]), axis=1))
        # q_mean = qs_padded.find_q_mean(q1)

        w_mean = np.sum(self.weights_mu * Y[4:], axis=1)

        mu_this_est = np.concatenate((q_mean.array.reshape(-1), w_mean.reshape(-1)))

        # Equations 65-67: Transform Y into W', notated as Wp for prime
        # rWp = utilities.normalize_vectors(q_mean.inverse().q_multiply(qs).to_vectors())
        rWp = q_mean.inverse().q_multiply(qs).to_vectors()
        wWp = Y[4:] - w_mean.reshape(-1, 1)
        Wp = np.concatenate((rWp, wWp))

        # Equation 64
        Pk_bar = np.matmul(Wp, Wp.T)
        Pk_bar /= W.shape[1]
        Pk_bar += self.Q

        # Equation 27 and 40
        gs_est = qs.rotate_vector(self.g_vector)
        Z = np.concatenate((gs_est, Y[4:]))

        # Equation 48
        # z_est = np.zeros(self.n)
        # z_est[3:] = np.mean(Z[3:], axis=1)
        # z_est[:3] = q_mean.rotate_vector(self.g_vector)
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
        correction = np.matmul(K, (z_this - z_est).reshape(-1, 1)).reshape(-1)
        q_correction = Quaternions.from_vectors(correction[:3])
        w_correction = correction[3:]

        # Equation 46
        mu_this = np.zeros(mu_this_est.shape)
        mu_this[:4] = Quaternions(mu_this_est[:4]).q_multiply(q_correction).array  #
        mu_this[4:] = mu_this_est[4:] + w_correction

        # Equation 75:
        P_this = Pk_bar - np.matmul(np.matmul(K, Pvv), K.T)

        self._debug_print(20, 20.1, np.round(K, 3))

        return mu_this, P_this


if __name__ == "__main__":
    from data.trainer import Trainer
    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--datanum", required=False, help="Number of data file (1 to 3 inclusive)")

    args = vars(parser.parse_args())

    # Noise parameters for UKF
    Rr = np.array([.05, .05, .15])
    Rw = np.array([.05 for _ in range(3)])
    R = np.identity(QuaternionUkf.n) * np.concatenate((Rr, Rw))
    Q = np.copy(R)
    # Q = np.identity(QuaternionUkf.n) * 4.5  # * 2.993 * 2
    # Q[:3, :3] *= 1
    # Q[3:, 3:] *= 10

    num = args["datanum"]
    if not num:
        planner = RoundTripPlanner()
        source = DataMaker(planner)
    else:
        source = DataStore(dataset_number=num, path_to_data="data")

    f = QuaternionUkf(source, R, Q)
    f.filter_data()

    print(f.free_param)

    if not num:
        utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.angles, f.angles)
    else:
        ImuFilter.plot_comparison(f.rots, f.ts_imu, source.rots_vicon, source.ts_vicon)

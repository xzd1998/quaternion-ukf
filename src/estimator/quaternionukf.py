"""
Quaternion UKF
^^^^^^^^^^^^^^
"""

import argparse

import numpy as np

from estimator.constants import STATE_DOF, NUM_AXES
from estimator.data import utilities
from estimator.data.datamaker import DataMaker
from estimator.data.datastore import DataStore
from estimator.data.trajectoryplanner import RoundTripPlanner
from estimator.state_estimator import StateEstimator
from estimator.quaternions import Quaternions


class QuaternionUkf(StateEstimator):

    g_vector = np.array([0, 0, 1])

    def __init__(self, source, R, Q, alpha=1, beta=2, kappa=2):

        super().__init__(source)

        self.R = R
        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Initialize covariance history and state history
        self.state = np.zeros((STATE_DOF + 1, self.imu_data.shape[-1]))
        self.state[:, 0] = np.array([1, 0, 0, 0, 0, 0, 0])
        self.covariance = np.zeros((STATE_DOF, STATE_DOF, self.state.shape[-1]))
        self.covariance[..., 0] = np.identity(STATE_DOF) * .01

        self._t = 0

        self.free_param = self.alpha ** 2 * (STATE_DOF + self.kappa) - STATE_DOF

        w0m = self.free_param / (STATE_DOF + self.free_param)
        qs_to_pad = 2 * STATE_DOF * w0m / (1 - w0m) - 1
        if qs_to_pad - np.round(qs_to_pad) < 1e-5:
            self.qs_to_pad = int(np.round(qs_to_pad))
        else:
            raise ValueError("Can't weight with fraction of quaternion to find quaternion mean")

        w0c = w0m + 1 - self.alpha ** 2 + self.beta
        wi = 1 / (2 * (STATE_DOF + self.free_param))

        self.weights_state = np.insert(np.ones(2 * STATE_DOF) * wi, 0, w0m)
        self.weights_cov = np.insert(np.ones(2 * STATE_DOF) * wi, 0, w0c)

    def _debug_print(self, t_min, t_max, *contents):
        if t_min <= self._t <= t_max:
            print("Time {} seconds".format(self._t))
            for content in contents:
                print(content)

    def _get_sigma_distances(self, cov_last):
        # m = STATE_DOF
        # S = np.linalg.cholesky(m * (cov_last + self.Q))
        # W = np.concatenate((S, -S), axis=1) / 10
        # return np.concatenate((np.zeros((STATE_DOF, 1)), W), axis=1)
        cov_muliplier = STATE_DOF
        positive_offsets = np.linalg.cholesky(cov_muliplier * (cov_last + self.Q))
        offsets = np.concatenate((positive_offsets, -positive_offsets), axis=1)
        return np.concatenate((np.zeros((STATE_DOF, 1)), offsets), axis=1)

    def _get_custom_distances(self):
        S = np.identity(STATE_DOF) * 3
        S[:3] /= 30
        W = np.concatenate((S, -S), axis=1)
        return np.concatenate((np.zeros((STATE_DOF, 1)), W), axis=1)

    def estimate_state(self):

        self.imu_data[:3] = self._normalize_data(self.imu_data[:3])

        rots = np.zeros((3, 3, self.state.shape[-1]))
        rots[..., 0] = Quaternions(self.state[:4, 0]).to_rotation_matrix()

        for i in range(1, self.state.shape[-1]):
            dt = self.ts_imu[i] - self.ts_imu[i - 1]
            self._t = self.ts_imu[i]

            self.state[:, i], self.covariance[..., i] = self._filter_next(
                self.covariance[..., i - 1],
                self.state[:, i - 1],
                self.imu_data[:, i],
                dt
            )
            rots[..., i] = Quaternions(self.state[:4, i]).to_rotation_matrix()

        self.rots = rots

    def _filter_next(self, P_last, mu_last, z_this, dt):

        W = self._get_sigma_distances(P_last)
        # W = self._get_custom_distances()

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
        # Y[:4] = Y[:4] * np.sign(Y[0])
        q1 = Quaternions(Y[:4, 0])
        qs = Quaternions(Y[:4])
        q_mean = qs.find_q_mean(q1)

        # extra_q1s = np.matmul(q1.array.reshape(-1, 1), np.ones((1, self.qs_to_pad)))
        # qs_padded = Quaternions(np.concatenate((extra_q1s, Y[:4]), axis=1))
        # q_mean = qs_padded.find_q_mean(q1)

        w_mean = np.sum(self.weights_state * Y[4:], axis=1)

        mu_this_est = np.concatenate((q_mean.array.reshape(-1), w_mean.reshape(-1)))

        # Equations 65-67: Transform Y into W', notated as Wp for prime
        # rWp = utilities.normalize_vectors(q_mean.inverse().q_multiply(qs).to_vectors())
        rWp = q_mean.inverse().q_multiply(qs).to_vectors()
        wWp = Y[4:] - w_mean.reshape(-1, 1)
        Wp = np.concatenate((rWp, wWp))

        # Equation 64
        Pk_bar = np.matmul(Wp, Wp.T)
        Pk_bar /= W.shape[1]
        # Pk_bar += self.Q

        # Equation 27 and 40
        gs_est = qs.rotate_vector(self.g_vector)
        Z = np.concatenate((gs_est, Y[4:]))

        # Equation 48
        # z_est = np.zeros(STATE_DOF)
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

        self._debug_print(0, .1, np.round(P_this, 3))

        return mu_this, P_this


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--datanum", required=False, help="Number of data file (1 to 3 inclusive)")

    args = vars(parser.parse_args())

    # Noise parameters for UKF
    # Rr = np.array([.05, .05, .15])
    # Rw = np.array([.05 for _ in range(3)])
    # R = np.identity(STATE_DOF) * np.concatenate((Rr, Rw))
    # Q = np.copy(R)
    # Q = np.identity(STATE_DOF) * 4.5574
    # Q[:3, :3] *= 1
    # Q[3:, 3:] *= 5
    R = np.identity(STATE_DOF) * .1
    # R[5, 5] = .001
    Q = np.copy(R)
    Q[3:, 3:] *= 10

    num = args["datanum"]
    if not num:
        planner = RoundTripPlanner()
        source = DataMaker(planner)
    else:
        source = DataStore(dataset_number=num, path_to_data="estimator/data/")

    f = QuaternionUkf(source, R, Q)
    f.estimate_state()

    print(f.free_param)

    if not num:
        utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.angles, f.angles)
    else:
        StateEstimator.plot_comparison(f.rots, f.ts_imu, source.rots_vicon, source.ts_vicon)

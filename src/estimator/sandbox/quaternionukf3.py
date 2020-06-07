import argparse

import numpy as np

from estimator.constants import NUM_AXES
from estimator.data import utilities
from estimator.data.datamaker import DataMaker
from estimator.data.datastore import DataStore
from estimator.data.trajectoryplanner import RoundTripPlanner
from estimator.state_estimator import StateEstimator
from estimator.quaternions import Quaternions


class QuaternionUkf3(StateEstimator):

    state_dof = 3
    g_vector = np.array([0, 0, 1])

    def __init__(self, source, R, Q, alpha=1, beta=2, kappa=2):

        super().__init__(source)

        self.R = R
        self.Q = Q
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Initialize covariance history and state history
        self.mu = np.zeros((NUM_AXES + 1, self.imu_data.shape[-1]))
        self.mu[:, 0] = np.array([1, 0, 0, 0])
        self.P = np.zeros((NUM_AXES, NUM_AXES, self.mu.shape[-1]))
        self.P[..., 0] = np.identity(NUM_AXES) * .05

        self._t = 0
        self.Wps = None
        self.iteration = None

    def _debug_print(self, t_min, duration, *contents):
        if t_min <= self._t <= t_min + duration:
            print("Time {} seconds".format(self._t))
            for content in contents:
                print(content)

    def _get_sigma_distances(self, P_last):
        m = NUM_AXES
        S = np.linalg.cholesky(m * (P_last + self.Q))
        W = np.concatenate((S, -S), axis=1) / 10
        return np.concatenate((np.zeros((NUM_AXES, 1)), W), axis=1)

    def estimate_state(self):

        self.imu_data[:3] = self._normalize_data(self.imu_data[:3])

        rots = np.zeros((3, 3, self.mu.shape[-1]))
        rots[..., 0] = Quaternions(self.mu[:4, 0]).to_rotation_matrix()

        self.Wps = np.zeros((3, 2 * NUM_AXES + 1, self.num_data))
        self.iteration = 0

        for i in range(1, self.mu.shape[-1]):
            self.iteration = i
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

        W = self._get_sigma_distances(P_last)

        # Equation 34: Form sigma points based on prior mean and covariance data
        qW = Quaternions.from_vectors(W)
        q_last = Quaternions(mu_last)
        q_sigpt = qW.q_multiply(q_last)

        # Equations 9-11: form q_delta
        qd = Quaternions.from_vectors(z_this[3:] * dt)

        # Equation 22: Apply non-linear function A with process noise of zero
        Y = qd.q_multiply(q_sigpt).array
        self._debug_print(12.15, 0.1, np.round(Y, 2))

        # Equations 52-55: Use mean-finding algorithm to satisfy Equation 38
        # Y = Y * np.sign(Y[0])
        q1 = Quaternions(Y[:, 0])
        qs = Quaternions(Y)
        q_mean = qs.find_q_mean(q1)

        mu_this_est = q_mean.array.reshape(-1)

        # Equations 65-67: Transform Y into W', notated as Wp for prime
        # Wp = utilities.normalize_vectors(q_mean.inverse().q_multiply(qs).to_vectors())
        Wp = q_mean.inverse().q_multiply(qs).to_vectors()
        self.Wps[..., self.iteration] = np.copy(Wp)

        # Equation 64
        Pk_bar = np.matmul(Wp, Wp.T)
        Pk_bar /= W.shape[1]

        # Equation 27 and 40
        Z = qs.rotate_vector(self.g_vector)

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
        correction = np.matmul(K, (z_this[:3] - z_est).reshape(-1, 1)).reshape(-1)
        q_correction = Quaternions.from_vectors(correction)

        # Equation 46
        mu_this = Quaternions(mu_this_est[:4]).q_multiply(q_correction).array  #

        # Equation 75:
        P_this = Pk_bar - np.matmul(np.matmul(K, Pvv), K.T)

        # self._debug_print(12.15, 0.1, np.round(Y, 2))

        return mu_this, P_this


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--datanum", required=False, help="Number of data file (1 to 3 inclusive)")

    args = vars(parser.parse_args())

    # Noise parameters for UKF
    R = np.identity(NUM_AXES) * .1
    # R[5, 5] = .001
    Q = np.copy(R)

    num = args["datanum"]
    if not num:
        planner = RoundTripPlanner()
        source = DataMaker(planner)
    else:
        source = DataStore(dataset_number=num, path_to_data="estimator/data/")

    f = QuaternionUkf3(source, R, Q)
    f.estimate_state()

    if not num:
        utilities.plot_rowwise_data(
            ["z-axis"],
            ["x", "y", "z"],
            [source.ts, source.ts],
            source.angles,
            f.angles
        )
        Wps = [f.Wps[:, i, :] for i in range(f.Wps.shape[1])]
        utilities.plot_rowwise_data(
            [i for i in range(f.Wps.shape[1])],
            ["x", "y", "z"],
            [source.ts for _ in range(f.Wps.shape[1])],
            *Wps
        )
        utilities.plot_rowwise_data(
            ["z-axis"],
            ["w", "i", "j", "k"],
            [source.ts for _ in range(f.mu.shape[1])],
            f.mu
        )
        pass
    else:
        StateEstimator.plot_comparison(f.rots, f.ts_imu, source.rots_vicon, source.ts_vicon)

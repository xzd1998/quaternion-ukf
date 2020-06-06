import numpy as np

from estimator.data.datamaker import DataMaker
from estimator.data import trajectoryplanner, utilities
from estimator.state_estimator import StateEstimator


class VectorUkf3(StateEstimator):

    g_vector = np.array([0, 0, 1]).reshape(-1, 1)
    state_dof = 3

    def __init__(self, source, R, Q):

        super().__init__(source)

        self.R = R
        self.Q = Q

        # Initialize covariance history and state history
        self.mu = np.zeros((self.n, self.num_data))

        self.P = np.zeros((self.n, self.n, self.num_data))
        self.P[..., 0] = np.identity(self.n) * .01

        # Keeps track of timestep for debugging
        self._t = 0

    def _get_sigma_distances(self, P_last):
        m = self.n
        S = np.linalg.cholesky(m * (P_last + self.Q))  #
        W = np.concatenate((S, -S), axis=1)
        return np.concatenate((np.zeros((self.n, 1)), W), axis=1)

    def _debug_print(self, t_min, duration, *contents):
        if t_min <= self._t <= t_min + duration:
            print("Time {} seconds".format(self._t))
            for content in contents:
                print(content)

    def estimate_state(self):
        self.imu_data[:3] = self._normalize_data(self.imu_data[:3])

        for i in range(1, self.mu.shape[-1]):
            dt = self.ts_imu[i] - self.ts_imu[i - 1]
            self._t = self.ts_imu[i]

            self.mu[:, i], self.P[..., i] = self._filter_next(
                self.P[..., i - 1],
                self.mu[:, i - 1],
                self.imu_data[:, i],
                dt
            )

        self.rots = utilities.vectors_to_rots(self.mu[:3])
        # for i in range(self.rots.shape[-1]):
        #     self.rots[..., i] = self.rots[..., i].T

    def _filter_next(self, P_last, mu_last, z_this, dt):

        W = self._get_sigma_distances(P_last)

        # Equation 34: Form sigma points based on prior mean and covariance data
        sigpts = mu_last.reshape(-1, 1) + W

        # Equation 22: Apply non-linear function A with process noise of zero
        Y = sigpts + z_this[3:].reshape(-1, 1) * dt

        mu_this_est = np.mean(Y, axis=1)

        # Equations 65-67: Transform Y into W', notated as Wp for prime
        Wp = Y - mu_this_est.reshape(-1, 1)

        # Equation 64
        Pk_bar = np.matmul(Wp, Wp.T)
        Pk_bar /= W.shape[1]

        # Equation 27 and 40
        gs_est = np.zeros((3, Y.shape[-1]))
        rs = utilities.vectors_to_rots(Y[:3])
        for i in range(Y.shape[-1]):
            gs_est[:, i] = (rs[..., i].T @ self.g_vector).reshape(-1)
        Z = gs_est

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

        # Equation 46
        mu_this = mu_this_est + correction

        # Equation 75:
        P_this = Pk_bar - np.matmul(np.matmul(K, Pvv), K.T)

        self._debug_print(10, 0.1, np.round(K.T, 2))

        return mu_this, P_this


if __name__ == "__main__":

    # Noise parameters for UKF
    R = np.identity(VectorUkf3.n) * .01
    # R[5, 5] = .001
    Q = np.copy(R)

    planner = trajectoryplanner.round_trip_easy
    source = DataMaker(planner)

    f = VectorUkf3(source, R, Q)
    f.estimate_state()

    utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [source.ts, source.ts], source.angles, f.angles)

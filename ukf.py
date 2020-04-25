'''
Filename: ukf.py
Author: Matt Lisle
Date: 02/19/19
Description: Unscented Kalman Filter for orientation tracking
'''

import numpy as np
import argparse
from scipy.io import loadmat
import matplotlib.pyplot as plt
from quaternion import Quaternion


class UKF(object):

    def __init__(self, data_num=1):
        self.num = data_num

        # Noise parameters for UKF
        n = 6
        Rr = np.array([.05, .05, .15])
        Rw = np.array([.01 for i in range(3)])
        self.R = np.identity(n) * np.concatenate((Rr, Rw))
        self.Q = np.identity(n) * 2.991
        self.Q[:3, :3] *= 2

        # Get parameters set up before we get to filtering the data
        imu = loadmat("imu/imuRaw" + str(self.num) + ".mat")

        # IMU data
        self.vals = imu["vals"].astype(float)
        self.t_imu = imu["ts"].reshape(-1)

        # Reshuffle gyro data to be back in x-y-z order
        temp = np.copy(self.vals)
        self.vals[3] = temp[4]
        self.vals[4] = temp[5]
        self.vals[5] = temp[3]

        # Initialize covariance history and state history
        self.mu = np.zeros((n + 1, self.vals.shape[-1]))
        self.mu[:, 0] = np.array([1, 0, 0, 0, 0, 0, 0])
        self.P = np.zeros((n, n, self.mu.shape[-1]))
        self.P[..., 0] = np.identity(n) * .5

    def run_ukf(self):

        self.rots = self.estimate_state()
        self.roll, self.pitch, self.yaw = UKF.R_to_angles(self.rots)
        self.make_plots()

    def estimate_state(self):

        # Hardcode learned parameters for acceleration/gyro since can't access training script in autograder
        mr = np.array([-0.09363796, -0.09438229, 0.09449341])
        br = np.array([47.88161084, 47.23512485, -47.39899347])
        mw = np.array([0.01546466, 0.01578361, 0.01610787])

        self.vals[:3, :] = self.vals[:3, :] * mr.reshape(3, 1) + br.reshape(3, 1)
        self.vals[3:, :] = self.vals[3:, :] * mw.reshape(3, 1)
        self.vals[3:, :] = self.vals[3:, :] - ((np.mean(self.vals[3:, :50], axis=1) +
                                                np.mean(self.vals[3:, -50:], axis=1)) / 2).reshape(3, 1)
        self.vals[:3] = self.vals[:3] / np.linalg.norm(self.vals[:3], axis=0)

        bot_rots = np.zeros((3, 3, self.mu.shape[-1]))
        bot_rots[..., 0] = Quaternion(self.mu[:4, 0]).q_to_rot()

        for i in range(1, self.mu.shape[-1]):
            dt = self.t_imu[i] - self.t_imu[i - 1]
            self.mu[:, i], self.P[..., i] = self.filter(self.P[..., i - 1], self.mu[:, i - 1], self.vals[:, i], dt)
            bot_rots[..., i] = Quaternion(self.mu[:4, i]).q_to_rot()

        return bot_rots

    def filter(self, P_last, mu_last, z_this, dt):

        # 2n sigma points
        n = P_last.shape[0]
        m = np.sqrt(n)
        S = np.linalg.cholesky(P_last + self.Q)
        W = np.concatenate((m * S, -m * S), axis=1)
        W = np.concatenate((np.zeros((n, 1)), W), axis=1)

        # Equation 34: Form sigma points based on prior mean and covariance data
        qW = Quaternion.v_to_q(W[:3])
        q_last = Quaternion(mu_last[:4])
        q_sigpt = q_last.multiply_q(qW)

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
        qd = Quaternion(qd.astype(float) / np.linalg.norm(qd, axis=0))

        # Equation 22: Apply non-linear function A with process noise of zero
        qY = q_sigpt.multiply_q(qd)
        Y = np.concatenate((qY.q, w_sigpt))

        # Equations 52-55: Use mean-finding algorithm to satisfy Equation 38
        Y[:4, :] = Y[:4, :] * np.sign(Y[0, :])
        q1 = Quaternion(Y[:4, 0])
        qs = Quaternion(Y[:4, :])
        q_mean = qs.find_q_mean(q1)
        w_mean = np.mean(Y[4:, :], axis=1)
        mu_this_est = np.concatenate((q_mean.q.reshape(-1), w_mean.reshape(-1)))

        # Equations 65-67: Transform Y into W', notated as Wp for prime
        rWp = q_mean.q_inv().multiply_q(qs).q_to_v()
        wWp = Y[4:, :] - w_mean.reshape(-1, 1)
        Wp = np.concatenate((rWp, wWp))

        # Equation 64
        Pk_bar = np.matmul(Wp, Wp.T)

        # Equation 27 and 40
        g = Quaternion(np.array([0, 0, 0, 1]))
        gp = qs.q_inv().multiply_q(g).multiply_q(qs)
        Z = np.concatenate((gp.q[1:], Y[4:, :]))

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
        q_this = Quaternion.v_to_q(r_this[:3])
        w_this = r_this[3:]

        # Equation 46
        mu_this = np.zeros(mu_this_est.shape)
        mu_this[:4] = Quaternion(mu_this_est[:4]).multiply_q(q_this).q
        mu_this[4:] = mu_this_est[4:] + w_this

        # Equation 75:
        P_this = Pk_bar - np.matmul(np.matmul(K, Pvv), K.T)

        return mu_this, P_this

    def make_plots(self):
        vicon = loadmat("vicon/viconRot" + str(self.num) + ".mat")
        imu = loadmat("imu/imuRaw" + str(self.num) + ".mat")

        R = vicon["rots"]
        t_vicon = vicon["ts"].reshape(-1)
        t_imu = imu["ts"].reshape(-1)
        t0 = min(t_vicon[0], t_imu[0])
        R = R[..., t_vicon > t_imu[0]]
        r = self.rots[..., t_imu > t_vicon[0]]

        labels = ["Roll", "Pitch", "Yaw"]

        a = UKF.R_to_angles(r)
        angs = UKF.R_to_angles(R)

        for i in range(3):
            plt.figure(i)
            plt.plot(t_vicon[t_vicon > t_imu[0]] - t0, angs[i])
            plt.plot(t_imu[t_imu > t_vicon[0]] - t0, a[i])
            plt.xlabel("Time [s]")
            plt.ylabel(labels[i] + " Angle [rad]")
            plt.grid(True)
            plt.legend(["Truth", "UKF"])
        plt.show()

    @staticmethod
    def R_to_angles(R):
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(np.square(R[2, 1]) + np.square(R[2, 2])))
        yaw = np.arctan2(R[1, 0], R[0, 0])

        return roll, pitch, yaw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-D", "--datanum", required=True, help="Number of data file (1 to 3 inclusive)")

    args = vars(parser.parse_args())

    f = UKF(args["datanum"])
    f.run_ukf()

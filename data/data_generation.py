
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from ukf import UKF

imu = loadmat("imu/imuRaw1.mat")
vals = imu["vals"].astype(float)
dt = 0.01  # seconds
T = 20  # seconds


class TrajectoryPlanner:
    def __init__(self):
        pass


def generate_data():
    ts = np.arange(0, T, dt)
    n = ts.shape[0]
    angs = np.zeros((n, 3))
    vels = np.zeros((n, 3))
    accs = np.zeros((n, 3))

    da = 1/4000

    def integrate(idx):
        vels[idx + 1] = vels[idx] + accs[idx] * dt
        angs[idx + 1] = angs[idx] + vels[idx] * dt + accs[idx] * dt ** 2 / 2

    for (i, t) in enumerate(ts[1:]):
        integrate(i)
        if 4 <= t < 7:
            accs[i + 1] = accs[i] + da
        if 7 <= t < 13:
            accs[i + 1] = accs[i] - da
        if 13 <= t < 16:
            accs[i + 1] = accs[i] + da

    return ts, angs.T, vels.T


def plot_saved_data(num):
    vicon = loadmat("vicon/viconRot" + str(num) + ".mat")
    imu = loadmat("imu/imuRaw" + str(num) + ".mat")

    R = vicon["rots"]
    t_vicon = vicon["ts"].reshape(-1)
    t_imu = imu["ts"].reshape(-1)
    t0 = min(t_vicon[0], t_imu[0])
    R = R[..., t_vicon > t_imu[0]]

    t = t_vicon[t_vicon > t_imu[0]] - t0
    angs = UKF.rots_to_angles(R)
    vels = np.diff(angs) / np.diff(t)

    plot_angle_data(t[:-1], vels)


def plot_angle_data(t, angs):
    labels = ["Roll", "Pitch", "Yaw"]
    for i in range(3):
        plt.figure(i)
        plt.plot(t, angs[i])
        plt.xlabel("Time [s]")
        plt.ylabel(labels[i] + " Angle [rad]")
        plt.grid(True)
    plt.show()


if __name__ == "__main__":
    ts, angs, vels = generate_data()
    roll, pitch, yaw = (angs[0], angs[1], angs[2])
    R = UKF.angles_to_rots(roll, pitch, yaw)
    angs_again = UKF.rots_to_angles(R)

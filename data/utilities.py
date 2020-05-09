import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g


def rots_to_angles(rots):
    roll = np.arctan2(rots[2, 1], rots[2, 2])
    pitch = np.arctan2(-rots[2, 0], np.sqrt(np.square(rots[2, 1]) + np.square(rots[2, 2])))
    yaw = np.arctan2(rots[1, 0], rots[0, 0])

    return roll, pitch, yaw


def rots_to_vels(rots, ts):
    roll, pitch, yaw = rots_to_angles(rots)
    make_angles_continuous([roll, pitch, yaw])
    dts = np.diff(ts)
    dr, dp, dy = (np.diff(roll) / dts, np.diff(pitch) / dts, np.diff(yaw) / dts)
    return np.vstack((dr, dp, dy)), ts[:-1]


def rots_to_accs(rots, noise=None):
    result = np.zeros((rots.shape[0], 1, rots.shape[-1]))
    gravity = np.array([0, 0, g]).reshape(3, 1)
    for i in range(rots.shape[-1]):
        result[..., i] = np.matmul(rots[..., i].T, gravity)
    result = result.reshape(3, -1)
    if noise is not None:
        result += noise
    return result.reshape(3, -1)


def make_angles_continuous(angles):
    for row in angles:
        for i in range(1, row.shape[0]):
            d = row[i] - row[i - 1]
            if np.abs(d) > np.pi:
                row[i] = row[i] - np.sign(d) * 2 * np.pi


def moving_average(data, n=9):
    averaged = np.zeros(data.shape)
    for i in range(data.shape[0]):
        averaged[i] = np.convolve(data[i], np.ones(n) / n, "same")
    return averaged


def angles_to_rots_zyx(roll, pitch, yaw):
    rots = np.zeros((3, 3, roll.shape[0]))
    rots[0, 0] = np.cos(yaw) * np.cos(pitch)
    rots[1, 0] = np.sin(yaw) * np.cos(pitch)
    rots[2, 0] = -np.sin(pitch)
    rots[0, 1] = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
    rots[1, 1] = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
    rots[2, 1] = np.cos(pitch) * np.sin(roll)
    rots[0, 2] = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
    rots[1, 2] = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
    rots[2, 2] = np.cos(pitch) * np.cos(roll)
    return rots


def angles_to_rots_xyz(roll, pitch, yaw):
    rots = np.zeros((3, 3, roll.shape[0]))
    rots[0, 0] = np.cos(yaw) * np.cos(pitch)
    rots[0, 1] = -np.sin(yaw) * np.cos(pitch)
    rots[0, 2] = np.sin(pitch)
    rots[1, 0] = np.cos(yaw) * np.sin(pitch) * np.sin(roll) + np.sin(yaw) * np.cos(roll)
    rots[1, 1] = -np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
    rots[1, 2] = np.cos(pitch) * np.sin(roll)
    rots[2, 0] = -np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
    rots[2, 1] = np.sin(yaw) * np.sin(pitch) * np.cos(roll) + np.cos(yaw) * np.sin(roll)
    rots[2, 2] = np.cos(pitch) * np.cos(roll)
    return rots


def plot_rowwise_data(data_labels, y_labels, ts, *datasets):
    n_plots = len(y_labels)
    n_sets = len(datasets)

    for dataset in datasets:
        if len(dataset) != n_plots:
            raise ValueError("Mismatch: {} labels versus {} rows of data".format(n_plots, n_sets))

    for i in range(n_plots):
        plt.figure(i)
        for j in range(n_sets):
            plt.plot(ts[j], datasets[j][i])
        plt.xlabel("Time [s]")
        plt.ylabel(y_labels[i] + " Angle [rad]")
        plt.legend(data_labels)
        plt.grid(True)
    plt.show()


def accs_to_roll_pitch(accs):
    roll = np.arctan2(accs[1], accs[2])
    pitch = np.arctan2(-accs[0], np.sqrt(np.square(accs[1]) + np.square(accs[2])))
    return roll, pitch


if __name__ == "__main__":
    rpy = np.random.randn(3, 1)
    print(rpy)
    rot = angles_to_rots_zyx(rpy[0], rpy[1], rpy[2])
    r, p, y = rots_to_angles(rot)
    rpy_again = np.array([r, p, y]).reshape(3, 1)
    print(rots_to_angles(rot))
    print(np.abs(rpy - rpy_again) < .001)


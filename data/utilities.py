import numpy as np
import matplotlib.pyplot as plt


def rots_to_angles(rots):
    roll = np.arctan2(rots[2, 1], rots[2, 2])
    pitch = np.arctan2(-rots[2, 0], np.sqrt(np.square(rots[2, 1]) + np.square(rots[2, 2])))
    yaw = np.arctan2(rots[1, 0], rots[0, 0])

    return roll, pitch, yaw


def angles_to_rots(roll, pitch, yaw):
    rots = np.zeros((3, 3, roll.shape[0]))
    rots[0, 0] = np.cos(yaw) * np.cos(pitch)
    rots[1, 0] = np.sin(yaw) * np.cos(pitch)
    rots[2, 0] = -np.sin(yaw)
    rots[0, 1] = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
    rots[1, 1] = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
    rots[2, 1] = np.cos(pitch) * np.sin(roll)
    rots[0, 2] = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
    rots[1, 2] = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
    rots[2, 2] = np.cos(pitch) * np.cos(roll)
    return rots


def plot_rowwise_data(data_labels, y_labels, ts, *datasets):
    n_plots = len(y_labels)
    n_sets = len(datasets)

    for dataset in datasets:
        if dataset.shape[0] != n_plots:
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

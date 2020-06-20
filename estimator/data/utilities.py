"""
Functions that don't fit nicely into any one class defined in the **estimator** module.
Most are conversions between different types of rotations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g


def rots_to_angles_zyx(rots):
    """
    Converts rotation matrices of shape (3, 3, N), N >= 0, to roll, pitch, and yaw
    Euler angles using the ZYX convention
    """
    roll = np.arctan2(rots[2, 1], rots[2, 2])
    pitch = np.arctan2(-rots[2, 0], np.sqrt(np.square(rots[2, 1]) + np.square(rots[2, 2])))
    yaw = np.arctan2(rots[1, 0], rots[0, 0])

    return roll, pitch, yaw


def rots_to_angles_xyz(rots):
    """
    Converts rotation matrices of shape (3, 3, N), N >= 0, to roll, pitch, and yaw
    Euler angles using the XYZ convention
    """
    roll = np.arctan2(-rots[1, 2], rots[2, 2])
    pitch = np.arctan2(rots[0, 2], np.sqrt(np.square(rots[1, 2]) + np.square(rots[2, 2])))
    yaw = np.arctan2(-rots[0, 1], rots[0, 0])

    return roll, pitch, yaw


def rots_to_vels(rots, time_vector):
    """
    Converts rotation matrices over time to angular velocities using the ZYX convention
    
    :param rots: (3, 3, N), N > 1 numpy array of rotation matrices
    :param time_vector: time vector of length N associated with rotation matrices
    :return: roll, pitch, and yaw velocities
    """
    smooth_factor = 8
    velocities = np.zeros((3, 1, rots.shape[-1] - smooth_factor))
    indexer = np.ones(velocities.shape[-1], dtype=bool)
    for i in range(smooth_factor, rots.shape[-1]):
        rot_diff = np.matmul(rots[..., i], rots[..., i - smooth_factor].T)
        theta = np.arccos((np.trace(rot_diff) - 1) / 2)
        if np.isnan(theta):
            indexer[i - smooth_factor] = False
        dt = time_vector[i] - time_vector[i - smooth_factor]
        denom = 2 * np.sin(theta)
        w_hat = ((theta / dt) / denom * (rot_diff - rot_diff.T)
                 if denom != 0 else np.zeros((3, 3)))

        # Got correspondence from first set of data
        velocities[0, 0, i - smooth_factor] = w_hat[2, 1]
        velocities[1, 0, i - smooth_factor] = w_hat[0, 2]
        velocities[2, 0, i - smooth_factor] = w_hat[1, 0]

    return velocities.reshape(3, -1), time_vector[smooth_factor // 2: -smooth_factor // 2]


def rots_to_accs(rots, noise=None):
    """
    Converts rotation matrices to the expected accelerometer data that would be gathered
    if the robot was in the provided series of rotations

    :param rots: (3, 3, N), N > 1 numpy array of rotation matrices
    :param noise: to apply to the accelerometer estimate, used when making toy data
    :return: expected accelerometer data
    """
    result = np.zeros((rots.shape[0], 1, rots.shape[-1]))
    gravity = np.array([0, 0, g]).reshape(3, 1)
    for i in range(rots.shape[-1]):
        result[..., i] = np.matmul(rots[..., i].T, gravity)
    result = result.reshape(3, -1)
    if noise is not None:
        result += noise
    return result.reshape(3, -1)


def moving_average(data, num=9):
    """
    Filter input data with moving average

    :param data: data to filter
    :param num: length of moving average window
    :return: averaged data
    """

    averaged = np.zeros(data.shape)
    for i in range(data.shape[0]):
        averaged[i] = np.convolve(data[i], np.ones(num) / num, "same")
    return averaged


def vectors_to_rots(raw):
    """
    Converts array of rotation vectors to rotation matrices

    :param raw: (3, N), N >= 0 numpy array of rotation vectors
    :return: corresponding rotation matrices
    """
    vecs = np.copy(raw)
    if len(vecs.shape) == 1:
        vecs = vecs.reshape(-1, 1)
    rots = np.zeros((3, 3, vecs.shape[-1]))
    axis = np.zeros((3, vecs.shape[-1]))
    theta = np.linalg.norm(vecs, axis=0)
    indexer = theta > 0

    axis[:, indexer] = vecs[:, indexer] / theta[indexer]
    axis_x = axis[0]
    axis_y = axis[1]
    axis_z = axis[2]

    c_th = np.cos(theta)
    v_th = 1 - np.cos(theta)
    s_th = np.sin(theta)

    rots[0, 0] = np.square(axis_x) * v_th + c_th
    rots[1, 0] = axis_x * axis_y * v_th + axis_z * s_th
    rots[2, 0] = axis_x * axis_z * v_th - axis_y * s_th
    rots[0, 1] = axis_x * axis_y * v_th - axis_z * s_th
    rots[1, 1] = np.square(axis_y) * v_th + c_th
    rots[2, 1] = axis_y * axis_z * v_th + axis_x * s_th
    rots[0, 2] = axis_x * axis_z * v_th + axis_y * s_th
    rots[1, 2] = axis_y * axis_z * v_th - axis_x * s_th
    rots[2, 2] = np.square(axis_z) * v_th + c_th

    return rots


def rots_to_vectors(raw):
    """
    Converts array of rotation matrices to rotation vectors

    :param raw: (3, 3, N), N >= 0 numpy array of rotation matrices
    :return: corresponding rotation vectors
    """
    if len(raw.shape) == 2:
        rots = raw.reshape(3, 3, 1)
    else:
        rots = raw

    vecs = np.zeros((3, rots.shape[-1]))

    thetas = np.arccos((rots[0, 0] + rots[1, 1] + rots[2, 2] - 1) / 2)
    valids = thetas != 0

    vecs[0, valids] = (rots[2, 1, valids] - rots[1, 2, valids]) / (2 * np.sin(thetas[valids]))
    vecs[1, valids] = (rots[0, 2, valids] - rots[2, 0, valids]) / (2 * np.sin(thetas[valids]))
    vecs[2, valids] = (rots[1, 0, valids] - rots[0, 1, valids]) / (2 * np.sin(thetas[valids]))

    return vecs * thetas


def angles_to_rots_zyx(roll, pitch, yaw):
    """
    Converts ZYX Euler angles to (3, 3, N) array of rotation matrices
    """

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
    """
    Converts XYZ Euler angles to (3, 3, N) array of rotation matrices
    """

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


def plot_data_comparison(data_labels, y_labels, time_vectors, datasets, rcparams=None, path=None):
    """
    Plots overlapping time series datasets together for comparison
    
    :param data_labels: labels for the legend for each dataset
    :param y_labels: labels for the y-axis of each plot
    :param time_vectors: time vector associated with each dataset
    :param datasets: data to plot
    :param rcparams: matplotlib parameters to change, e.g., fontsize
    :param path: where to save the figures once generated
    """

    if rcparams:
        plt.rcParams.update(rcparams)

    n_plots = len(y_labels)
    n_sets = len(datasets)

    for dataset in datasets:
        if len(dataset) != n_plots:
            raise ValueError("Mismatch: {} labels versus {} rows of data".format(n_plots, n_sets))

    for i in range(n_plots):
        plt.figure(i)
        for j in range(n_sets):
            plt.plot(time_vectors[j], datasets[j][i])
        plt.xlabel("Time [s]")
        plt.ylabel(y_labels[i] + " Angle [rad]")
        plt.legend(data_labels)
        plt.grid(True)
        if path:
            # manager = plt.get_current_fig_manager()
            # manager.resize(*manager.window.maxsize())
            fig = plt.gcf()
            fig.set_size_inches(16, 9)
            plt.savefig("{}_{}.png".format(path, y_labels[i].lower()),
                        bbox_inches='tight')
    plt.show()


def accs_to_roll_pitch(accs):
    """
    Calculates roll and pitch from accelerometer data assuming ZYX convention

    :param accs: (3, N), N >= 0 numpy array of accelerometer data
    :return: corresponding arrays of roll and pitch angles
    """

    roll = np.arctan2(accs[1], accs[2])
    pitch = np.arctan2(-accs[0], np.sqrt(np.square(accs[1]) + np.square(accs[2])))
    return roll, pitch


def normalize_vectors(vectors):
    """Normalizes vectors row-wise, ignoring vectors of all zeros."""

    norm = np.linalg.norm(vectors, axis=0)
    vectors[:, norm > 0] /= norm[norm > 0]
    return vectors

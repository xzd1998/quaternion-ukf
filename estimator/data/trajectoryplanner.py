"""
Holds a few different types of trajectory planners which are useful for testing different
aspects of the estimator.

* :code:`TrajectoryPlanner`: parent class for planners which isn't instantiated directly
* :code:`SimplePlanner`: plans a one-directional, minimum force trajectory from zero to
  some other orientation
* :code:`RoundTripPlanner`: plans a two-directional, minimum force trajectory from zero to
  some other orientation back to zero
* :code:`StationaryPlanner`: plans a zero-movement trajectory
"""

import numpy as np


class TrajectoryPlanner:
    """
    Parent class for trajectory planners which defines the interface to plan minimum force
    trajectories among different orientations

    Note on the noise models: I decided to not apply any drift to the accelerometer data
    because I found that most helpful when testing, but :code:`noise_stddev` and
    :code:`drift_stddev` don't _have_ to be applied to accelerometer and gyro data,
    respectively

    :ivar duration: of the data sample in seconds
    :ivar dt: time increment of the data
    :ivar noise_stddev: standard deviation of the white noise to add to accelerometer data
    :ivar drift_stddev: standard deviation of the brownian motion to add to gyro data
    :ivar bounds: time boundaries of the format [start, end), in seconds, which delineate
                  which time segments apply to which acceleration calculation
    :ivar acc_calculators: functions that, given no arguments, tell the trajectory planner
                           what acceleration to use
    """

    def __init__(self, duration, dt, noise_stddev, drift_stddev, bounds, *acc_calculators):

        self.duration = duration
        self.dt = dt
        num_data = int(duration / dt)

        self.noise_stddev = noise_stddev
        self.drift_stddev = drift_stddev

        if len(bounds) != len(acc_calculators):
            raise ValueError("Mismatch: {} bounds for {} calculators"
                             .format(len(bounds), len(acc_calculators)))
        if not all([len(pair) == 2 for pair in bounds]):
            raise ValueError("Bounds must be pairs of upper/lower bounds, but got: {}"
                             .format(bounds))

        np.random.seed(0)
        self.noise = np.random.randn(num_data) * noise_stddev
        self.drift = np.cumsum(np.random.randn(num_data) * drift_stddev)

        for (i, bound) in enumerate(bounds[:-1]):
            others = bounds[i + 1:]
            for other in others:
                lower_intersects = other[0] <= bound[0] < other[1]
                upper_intersects = other[0] < bound[1] <= other[1]
                if lower_intersects or upper_intersects:
                    print(lower_intersects)
                    print(upper_intersects)
                    raise ValueError("Intersecting bounds were provided: {}".format(bounds))

        self.bounds = bounds
        self.calculator_map = dict(zip(bounds, acc_calculators))

    def get_calculator(self, bound):
        """Returns the active acceleration calculator for a given time boundary"""
        return self.calculator_map[bound]

    @staticmethod
    def incrementer(amount):
        """Accelration calculator that increments by some amount"""
        return lambda x: x + amount

    @staticmethod
    def decrementer(amount):
        """Accelration calculator that decrements by some amount"""
        return lambda x: x - amount


class SimplePlanner(TrajectoryPlanner):
    """
    Plans a one-directional, minimum force trajectory from zero to some other orientation
    """
    def __init__(self, noise_stddev=0.02, drift_stddev=0.002):
        duration = 20
        dt = 0.01
        acc_increment = 1 / 3450 * np.array([1, 1, 0])
        super().__init__(
            duration,
            dt,
            noise_stddev,
            drift_stddev,
            ((4, 7), (7, 13), (13, 16)),
            TrajectoryPlanner.incrementer(acc_increment),
            TrajectoryPlanner.decrementer(acc_increment),
            TrajectoryPlanner.incrementer(acc_increment)
        )


class RoundTripPlanner(TrajectoryPlanner):
    """
    Plans a two-directional, minimum force trajectory from zero to some other orientation
    back to zero
    """
    def __init__(self, acc_magnitude=0.0005, noise_stddev=0.02, drift_stddev=0.002):
        duration = 30
        dt = 0.01
        acc_increment = acc_magnitude * np.array([1, 1, 0])
        super().__init__(
            duration,
            dt,
            noise_stddev,
            drift_stddev,
            ((4, 7), (7, 13), (13, 16), (16, 19), (19, 25), (25, 28)),
            TrajectoryPlanner.incrementer(acc_increment),
            TrajectoryPlanner.decrementer(acc_increment),
            TrajectoryPlanner.incrementer(acc_increment),
            TrajectoryPlanner.decrementer(acc_increment),
            TrajectoryPlanner.incrementer(acc_increment),
            TrajectoryPlanner.decrementer(acc_increment)
        )


class StationaryPlanner(TrajectoryPlanner):
    """
    Plans a zero-movement trajectory
    """
    def __init__(self):
        duration = 60
        dt = 0.01
        super().__init__(duration, dt, 0, 0, [])

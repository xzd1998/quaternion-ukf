import numpy as np


class TrajectoryPlanner:

    def __init__(self, duration, dt, noise_stddev, drift_stddev, bounds, *acc_calculators):

        self.duration = duration
        self.dt = dt
        self.num_data = int(duration / dt)

        if len(bounds) != len(acc_calculators):
            raise ValueError("Mismatch: {} bounds for {} calculators".format(len(bounds), len(acc_calculators)))
        if not all([len(pair) == 2 for pair in bounds]):
            raise ValueError("Bounds must be pairs of upper/lower bounds, but got: {}".format(bounds))
        # if duration % dt != 0:
        #     raise ValueError("Duration {} s not divisible by time increment {} s".format(duration, dt))

        np.random.seed(0)
        self.noise = np.random.randn(self.num_data) * noise_stddev
        self.drift = np.cumsum(np.random.randn(self.num_data) * drift_stddev)

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
        self.calculator_map = {bound: calculator for (bound, calculator) in zip(bounds, acc_calculators)}

    def get_calculator(self, bound):
        return self.calculator_map[bound]

    @staticmethod
    def incrementer(d):
        return lambda x: x + d

    @staticmethod
    def decrementer(d):
        return lambda x: x - d


class SimplePlanner(TrajectoryPlanner):

    def __init__(self, noise_stddev=0.1, drift_stddev=0.0004):
        duration = 20
        dt = 0.01
        da = 1 / 3450 * np.array([1, 1, 0])
        super().__init__(
            duration,
            dt,
            noise_stddev,
            drift_stddev,
            ((4, 7), (7, 13), (13, 16)),
            TrajectoryPlanner.incrementer(da),
            TrajectoryPlanner.decrementer(da),
            TrajectoryPlanner.incrementer(da)
        )


class RoundTripPlanner(TrajectoryPlanner):

    def __init__(self, noise_stddev=0.1, drift_stddev=0.0004):
        duration = 30
        dt = 0.01
        da = 1 / 2000 * np.array([1, 1, 0])
        super().__init__(
            duration,
            dt,
            noise_stddev,
            drift_stddev,
            ((4, 7), (7, 13), (13, 16), (16, 19), (19, 25), (25, 28)),
            TrajectoryPlanner.incrementer(da),
            TrajectoryPlanner.decrementer(da),
            TrajectoryPlanner.incrementer(da),
            TrajectoryPlanner.decrementer(da),
            TrajectoryPlanner.incrementer(da),
            TrajectoryPlanner.decrementer(da)
        )


class StationaryPlanner(TrajectoryPlanner):

    def __init__(self):
        duration = 60
        dt = 0.01
        super().__init__(duration, dt, 0, 0, [])

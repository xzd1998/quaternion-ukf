

class TrajectoryPlanner:

    def __init__(self, duration, dt, bounds, *acc_calculators):

        self._duration = duration
        self._dt = dt

        if len(bounds) != len(acc_calculators):
            raise ValueError("Mismatch: {} bounds for {} calculators".format(len(bounds), len(acc_calculators)))
        if not all([len(pair) == 2 for pair in bounds]):
            raise ValueError("Bounds must be pairs of upper/lower bounds, but got: {}".format(bounds))

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

    @property
    def duration(self):
        return self._duration

    @property
    def dt(self):
        return self._dt

    @staticmethod
    def incrementer(d):
        return lambda x: x + d

    @staticmethod
    def decrementer(d):
        return lambda x: x - d


class SimplePlanner(TrajectoryPlanner):

    def __init__(self):
        duration = 20
        dt = 0.01
        da = 1 / 3450
        super().__init__(
            duration,
            dt,
            ((4, 7), (7, 13), (13, 16)),
            TrajectoryPlanner.incrementer(da),
            TrajectoryPlanner.decrementer(da),
            TrajectoryPlanner.incrementer(da)
        )

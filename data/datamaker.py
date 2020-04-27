import numpy as np
from scipy.constants import g

import utilities


class DataMaker:

    def __init__(self, T, dt, planner):

        if T < dt:
            raise ValueError("Total time can't be less than the increment")

        self.ts = np.arange(0, T, dt)
        n = self.ts.shape[0]
        angs = np.zeros((n, 3))
        vels = np.zeros((n, 3))
        accs = np.zeros((n, 3))

        def integrate(idx):
            vels[idx + 1] = vels[idx] + (accs[idx + 1] + accs[idx]) * dt / 2
            angs[idx + 1] = angs[idx] + (vels[idx + 1] + vels[idx]) * dt / 2

        for (i, t) in enumerate(self.ts[:-1]):
            for bound in planner.bounds:
                if bound[0] <= t < bound[1]:
                    calculator = planner.get_calculator(bound)
                    accs[i + 1] = calculator(accs[i])
            integrate(i)

        self.angs = angs.T
        self.vels = vels.T
        self.rots = utilities.angles_to_rots(self.angs[0], self.angs[1], self.angs[2])

    @property
    def zs(self):
        return self.rots[:, -1]

    @property
    def gs(self):
        return self.zs * np.array([-g, -g, g]).reshape(3, 1)


class TrajectoryPlanner:

    def __init__(self, bounds, *acc_calculators):

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

    @staticmethod
    def incrementer(d):
        return lambda x: x + d

    @staticmethod
    def decrementer(d):
        return lambda x: x - d


if __name__ == "__main__":
    T = 20
    dt = 0.01
    da = 1 / 4000
    planner = TrajectoryPlanner(
        ((4, 7), (7, 13), (13, 16)),
        TrajectoryPlanner.incrementer(da),
        TrajectoryPlanner.decrementer(da),
        TrajectoryPlanner.incrementer(da)
    )
    maker = DataMaker(20, 0.01, planner)
    utilities.plot_rowwise_data(["z-axis"], ["x", "y", "z"], [maker.ts], maker.angs)

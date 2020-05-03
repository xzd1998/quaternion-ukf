import numpy as np
from scipy.constants import g

from data import utilities
from data.trajectoryplanner import *


class DataMaker:

    def __init__(self, planner):

        if planner.duration < planner.dt:
            raise ValueError("Total time can't be less than the increment")

        self.ts = np.arange(0, planner.duration, planner.dt)
        n = self.ts.shape[0]
        angs = np.zeros((n, 3))
        vels = np.zeros((n, 3))
        accs = np.zeros((n, 3))

        def integrate(idx):
            vels[idx + 1] = vels[idx] + (accs[idx + 1] + accs[idx]) * planner.dt / 2
            angs[idx + 1] = angs[idx] + (vels[idx + 1] + vels[idx]) * planner.dt / 2

        for (i, t) in enumerate(self.ts[:-1]):
            for bound in planner.bounds:
                if bound[0] <= t < bound[1]:
                    calculator = planner.get_calculator(bound)
                    accs[i + 1] = calculator(accs[i])
            integrate(i)

        self.angs_g = angs.T
        self.vels_g = vels.T

        # Rotation of the robot frame with respect to the global frame
        self.rots_g = utilities.angles_to_rots(self.angs_g[0], self.angs_g[1], self.angs_g[2])

        # Rotation of the global frame with respect to the robot frame
        self.rots_r = np.zeros(self.rots_g.shape)
        self.vels_r = np.zeros(self.vels_g.shape)
        for i in range(self.rots_g.shape[-1]):
            self.rots_r[..., i] = self.rots_g[..., i].T
            self.vels_r[..., i] = np.matmul(self.rots_r[..., i], self.vels_g[..., i].reshape(3, 1)).reshape(-1)

    @property
    def zs(self):
        return self.rots_g[:, -1]

    @property
    def gs(self):
        result = np.zeros((self.rots_g.shape[0], 1, self.rots_g.shape[-1]))
        gravity = np.array([0, 0, g]).reshape(3, 1)
        for i in range(self.rots_g.shape[-1]):
            result[..., i] = np.matmul(self.rots_r[..., i], gravity)
        return result.reshape(3, -1)

    @property
    def t_imu(self):
        return self.ts

    @property
    def t_vicon(self):
        return self.ts

    @property
    def vals(self):
        np.random.seed(1)
        noise = np.random.randn(self.ts.shape[0]) * 0.000
        return np.concatenate((self.gs, self.vels_g), axis=0) + noise


if __name__ == "__main__":
    planner = StationaryPlanner()
    maker = DataMaker(planner)
    print(maker.angs_g)
    ang_labels = ["roll", "pitch", "yaw"]
    vel_labels = ["wx", "wy", "wz"]
    utilities.plot_rowwise_data(["z-axis"], ang_labels + vel_labels, [maker.ts], maker.vals)

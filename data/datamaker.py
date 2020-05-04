import numpy as np
from scipy.constants import g

from data import utilities
from data.datasource import DataSource
from data.trajectoryplanner import *


class DataMaker(DataSource):
    """
    Creates test data from a pre-planned trajectory
    """
    def __init__(self, planner):
        """
        :param planner: defines what the angular acceleration is at each timestep
        """
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

        self.angs_vicon = angs.T
        vels_imu = vels.T
        vels_imu += planner.drift

        # Rotation of the robot frame with respect to the global frame
        rots_vicon = utilities.angles_to_rots(self.angs_vicon[0], self.angs_vicon[1], self.angs_vicon[2])
        accs = utilities.rots_to_accs(rots_vicon, planner.noise)

        super().__init__(self.ts, rots_vicon, self.ts, accs, vels_imu)


if __name__ == "__main__":
    planner = SimplePlanner()
    maker = DataMaker(planner)
    print(maker.angs_vicon)
    ang_labels = ["roll", "pitch", "yaw"]
    vel_labels = ["wx", "wy", "wz"]
    utilities.plot_rowwise_data(["z-axis"], ang_labels + vel_labels, [maker.ts], maker.data_imu)

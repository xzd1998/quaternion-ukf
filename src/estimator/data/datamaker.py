import numpy as np

from estimator.data.datasource import DataSource
from estimator.data import utilities
from estimator.data.trajectoryplanner import SimplePlanner


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

        self.vectors = angs.T
        vel_data = vels.T
        vel_data += planner.drift

        # Rotation of the robot frame with respect to the global frame
        rots_vicon = utilities.vectors_to_rots(self.vectors)
        acc_data = utilities.rots_to_accs(rots_vicon, planner.noise)

        super().__init__(self.ts, rots_vicon, self.ts, acc_data, vel_data)


if __name__ == "__main__":
    planner = SimplePlanner()
    maker = DataMaker(planner)
    ang_labels = ["roll", "pitch", "yaw"]
    vel_labels = ["wx", "wy", "wz"]
    utilities.plot_rowwise_data(["z-axis"], ang_labels, [maker.ts], maker.angles)
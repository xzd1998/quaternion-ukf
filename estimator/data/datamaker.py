"""
Builds toy data for testing the various estimator implementations. This is useful
because it's difficult to know if your model for process and measurement noise is
correct, but when you make the data, you determine the model and, therefore, reduce
your unknowns when debugging the UKF implementation.

In addition, you can build your own trajectories to test particular points that
may demonstrate instabilities in your estimation. For me, this included multiples
of :code:`pi/2`.

To build some toy data, a trajectory planner needs to be supplied, which the data
maker will then use to back calculate what the gyro and accelerometer readings
_would_ be for the planned trajectory. For more on how to define a trajectory
planner, see :mod:`data.trajectoryplanner`.

Here's an example to instantiate a :code:`DataMaker` and plot the orientation data
that it makes:

.. code-block::
    :linenos:

    planner = SimplePlanner()
    maker = DataMaker(planner)
    ang_labels = ["Roll", "Pitch", "Yaw"]
    utilities.plot_data_comparison(
        ["data"],
        ang_labels,
        [maker.ts_data],
        [maker.angles]
    )
"""

import numpy as np

from estimator.data.datasource import DataSource
from estimator.data import utilities


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

        ts_data = np.arange(0, planner.duration, planner.dt)
        num_data = ts_data.shape[0]
        angs = np.zeros((num_data, 3))
        vels = np.zeros((num_data, 3))
        accs = np.zeros((num_data, 3))

        def integrate(idx):
            vels[idx + 1] = vels[idx] + (accs[idx + 1] + accs[idx]) * planner.dt / 2
            angs[idx + 1] = angs[idx] + (vels[idx + 1] + vels[idx]) * planner.dt / 2

        for (i, t) in enumerate(ts_data[:-1]):
            for bound in planner.bounds:
                if bound[0] <= t < bound[1]:
                    calculator = planner.get_calculator(bound)
                    accs[i + 1] = calculator(accs[i])
            integrate(i)

        ang_data = angs.T
        vel_data = vels.T
        vel_data += planner.drift

        # Rotation of the robot frame with respect to the global frame
        rots_vicon = utilities.vectors_to_rots(ang_data)
        acc_data = utilities.rots_to_accs(rots_vicon, planner.noise)

        super().__init__(ts_data, rots_vicon, ts_data, acc_data, vel_data)

    def plot_data(self):
        """
        Plots the data made by this DataMaker
        """

        ang_labels = ["Roll", "Pitch", "Yaw"]
        utilities.plot_data_comparison(
            ["data"],
            ang_labels,
            [self.ts_imu],
            [self.angles_vicon]
        )

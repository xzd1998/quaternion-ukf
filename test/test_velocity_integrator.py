import unittest

import numpy as np

from estimator.data.datamaker import DataMaker
from estimator.data.trajectoryplanner import RoundTripPlanner
from estimator.velocity_integrator import VelocityIntegrator


class VelocityIntegratorTest(unittest.TestCase):

    def test_velocity_no_noise(self):
        planner = RoundTripPlanner(acc_magnitude=0.0005, noise_stddev=0, drift_stddev=0)
        data_source = DataMaker(planner)
        integrator = VelocityIntegrator(data_source)
        integrator.estimate_state()
        np.testing.assert_array_almost_equal(integrator.rots, data_source.rots_vicon, 2)

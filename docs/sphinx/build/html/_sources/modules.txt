estimator
=========

.. toctree::
   :maxdepth: 4

   estimator

Module contents
---------------

.. automodule:: estimator
   :members:
   :undoc-members:
   :show-inheritance:

Running this module runs a demo that plots a comparison of the UKF, the result of just integrating the gyro data, and the result of calculating the roll and pitch from the acceleromter data correctly. See the README for why the yaw can't be calculated directly.

To run the estimators with toy data, you can use the following:

.. code-block:: bash

    make run

To run the estimators on real data from an IMU, specify a dataset number from 1 to 3:

.. code-block:: bash

    make run DATASET=<n>

Where :code:`<n>` is a number from 1 to 3.

These datasets were provided by the instructors as part of the project. The output of the demo are the graphs of roll, pitch, and yaw, and the RMSE of each angle is printed to stdout.

For a well-tuned filter, the RMSE of the UKF will beat (be lower than) that of integrating the gyro data **or** calculating roll and pitch from the accelerometer.

This is evident when you look at the results of running the estimators on the toy data, which has the following output

.. code-block::

    RollPitchCalculator RMSE: [0.0019, 0.0024, 0.9610]
    VelocityIntegrator RMSE:  [0.6236, 0.4494, 0.3113]
    QuaternionUkf RMSE:       [0.0013, 0.0017, 0.1045]

Even though the gyro data has a **lot** of drift, the UKF is able to fuse the sensor data into a more accurate estimate than if one sensor had been trusted completely.

Getting a good model of the noise for the toy data is easy because I defined the model, but the same isn't true for the real IMU data. I've made my best guess for the process and measurement noise by looking at the coefficient of determination from calibrating the data.

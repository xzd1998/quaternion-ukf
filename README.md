---
# Quaternion UKF


Implentation of an unscented Kalman filter for orientation tracking of a
robot (e.g. a quadrotor or drone).

Introduction
---

This implementation of a UKF for tracking orientation of a drone with
gyro and accelerometer data follows closely that described in the paper
"A Quaternion-based Unscented Kalman Filter for Orientation Tracking" by
Edgar Kraft.

This project was completed as part of ESE 650: Learning in Robotics at
the University of Pennsylvania, though it has been tweaked, and
additional infrastructure has been built since then. This additional
infrastructure includes a means to manufacture toy data to validate that
the filter is indeed working properly.

Project Setup
---

-   Clone repository

``` 
git clone https://github.com/mattlisle/quaternion-ukf.git
```

-   Create a virtual environment and run setup script (python version
    should be &gt;= 3.6)

``` 
virtualenv --python=/usr/bin/python /path/to/new/virtualenv
source /path/to/new/virtualenv/bin/activate
make install
```

Usage
---

There are three sample datasets to run the code on. To run in terminal:

``` 
make run DATASET=<n>
```

Where `<n>` is a number from 1 to 3.

Implementation Details
---

As mentioned above, the implementation follows the example in the paper
fairly closely. The goal is to fuse accelerometer and gyro data to get
the best possible estimate of orientation, in terms of roll, pitch, and
yaw.

Sensor data processing
----------------------

The first step in this process is to convert the IMU data from some
digital reading to the corresponding value with the correct units. These
sensors are manufactured to be fairly linear in the range in which the
drone operates, so a linear regression model is a reasonable fit for
calibrating the data, which leads to the following equation:

$$\boldsymbol{\beta} = \left(\textbf{A}^{\top}\textbf{A}\right)^{-1}\textbf{A}^{\top}\textbf{y}$$

Where:

-   $\boldsymbol{\beta}$ represents a vector made of essentially the
    slope and intercept for fitting the IMU data (which is what
    we want).
-   **A** has two columns, the first is one dimension of the IMU
    data, e.g. the accelerometer's x-component
-   **y** is the ground truth, which we don't quite have yet}

Okay, so we've got the equation, (which numpy implements for us with
`np.linalg.lstsq` ), but there's still the unknown of how
to get the ground truth data to fit the IMU data to. That's where the
vicon data comes in, which is a time series of rotation matrices over
approximately the same timespan as the IMU data. Getting the expected
accelerometer data is quite straightforward because the accelerometer is
a measurement of the acceleration felt along the robot's z-axis. In
equation form:

$$\hat{\textbf{g}}^r = \textbf{R}_w^r \hat{\textbf{g}}^w$$

Gravity in the world frame is just the world frame's z-axis, and the
vicon rotations define the rotation between the world and robot frames.
However, solving for the rotational velocity is a bit trickier. If we
assume that the angular velocity is constant, then we can say the
following:

$$\dot{\textbf{R}} &= \textbf{S}(\boldsymbol{\omega})\textbf{R} \\
\textbf{R}(t) &= \exp \Big( \textbf{S}(\boldsymbol{\omega}) \Delta t \Big) \textbf{R}_0 \\
\textbf{R}_1 \textbf{R}_0^{\top} &= \exp \Big( \textbf{S}(\boldsymbol{\omega}) \Delta t \Big) = R$$

The left side of that equation represents the rotation in a given
timestep, which can be represented by an axis and an angle:

$$\theta &= \cos^{-1} \left( \frac{\text{Tr}(R) - 1}{2} \right) \\
\textbf{S}(\textbf{u}) &= \frac{1}{2 \sin \theta} \left( R - R^{\top} \right)$$

We can also convert from axis angle directly to angular velocity, which
leads us to a solution we can actually use:

$$\boldsymbol{\omega} &= \frac{\theta}{\Delta t} \textbf{u} \\
\textbf{S}(\boldsymbol{\omega}) &= \frac{\theta}{2 \Delta t \sin \theta} \left( R - R^{\top} \right)$$

So now we have the following relationship where everything on the right
side is known.

$$\boldsymbol{\beta}_a &= \left(\textbf{A}^{\top}\textbf{A}\right)^{-1}\textbf{A}^{\top}\textbf{g}^r \\
\boldsymbol{\beta}_\omega &= \left(\textbf{A}^{\top}\textbf{A}\right)^{-1}\textbf{A}^{\top}\boldsymbol{\omega}$$

And from these equations we can convert the accelerometer and gyro data
to units of acceleration and angular velocity, respectively

#!/usr/bin/env python
from setuptools import setup

requirements = [
  "numpy==1.22.0",
  "scipy==1.4.1",
  "matplotlib==3.0.3",
  "lazy==1.4",
]

setup(
    name="quaternionukf",
    python_requires=">=3.5",
    install_requires=requirements,
)
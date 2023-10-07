#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install numpy and matplotlib
pip install numpy matplotlib

# Install pybullet
pip install pybullet

# Install pytorch-mppi
pip install pytorch-mppi

# Install torch and torchvision
pip install torch torchvision 

# Install torchdiffeq
pip install torchdiffeq

echo "All dependencies installed successfully!"

#!/bin/bash

# Install pyarmor
pip install pyarmor==6.7.4

# Set the LD_LIBRARY_PATH environment variable
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/pytransform

# Run the checker_client script
python -u checker_client.py

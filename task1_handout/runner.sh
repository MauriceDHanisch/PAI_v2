#!/bin/bash

# Build the Docker image
docker build --tag task1 .

# Generate a unique filename based on the current date and time
timestamp=$(date +%m.%d._%Hh%M)
log_file="Dockerlog$timestamp.txt"

# Run the Docker container and save the logs to the file
#docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task1
docker run --rm -it -v "$( cd "$( dirname "$0" )" && pwd )":/results task1 /bin/bash 2>&1 | tee $log_file
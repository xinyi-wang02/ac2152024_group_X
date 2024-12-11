#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="215_data_pipeline_tensor"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../github_no_ship
export LOCAL_MOUNT_DIR=$(pwd)/../../../github_no_ship
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/data-service-account-model.json"

# Build the image based on the Dockerfile
docker build --no-cache -t $IMAGE_NAME -f Dockerfile .
#docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .

# Run the container
docker run --rm \
--name $IMAGE_NAME \
-e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
-v "$LOCAL_MOUNT_DIR":/no_ship \
-v "$SECRETS_DIR":/secrets \
-v "./":/app \
-ti $IMAGE_NAME

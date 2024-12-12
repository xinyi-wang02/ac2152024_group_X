#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="215_data_pipeline_mini"
export BASE_DIR=$(pwd)
export LOCAL_MOUNT_DIR="./no_ship"
export SECRETS_DIR="./secrets"

# Build the image based on the Dockerfile
#docker build -t $IMAGE_NAME -f Dockerfile .
docker build --no-cache -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .

# Run the container
docker run --rm \
--name $IMAGE_NAME \
-e GOOGLE_APPLICATION_CREDENTIALS='/secrets/data-service-account.json' \
-v "$SECRETS_DIR":/secrets \
-ti $IMAGE_NAME

# -v $LOCAL_MOUNT_DIR:/no_ship \

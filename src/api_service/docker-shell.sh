#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="215_data_pipeline_model_deployment_api"
export BASE_DIR=$(pwd)
export LOCAL_MOUNT_DIR="./no_ship"

# Build the image based on the Dockerfile
docker build --no-cache -t $IMAGE_NAME -f Dockerfile .
#docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .

# Run the container
docker run --rm \
--name $IMAGE_NAME \
-e GOOGLE_APPLICATION_CREDENTIALS='/no_ship/ai-service-account.json' \
-v $LOCAL_MOUNT_DIR:/no_ship \
-v "./":/app \
-p 9000:9000 \
-ti $IMAGE_NAME

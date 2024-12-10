#### Tensorizing container (image_train_preparation)

-   This container downloads preprocessed images and their labels from the GCP Bucket and convert the images to tensors and save them as `.tfrecord` in another GCS Bucket.
-   Input to this container is the GCP file location of the preprocessed image folder and the destination GCS Bucket that saves the tensorized data, secrets needed - via docker
-   Output from this container stored at GCS Bucket

(1)`src/image_train_preparation/tensorizing.py` - Here we download images and a CSV file from a specified Google Cloud Storage (GCS) bucket, processes the images by resizing them to 224x224 pixels, and serializes the images and labels into a TFRecord file. Then, we upload the resulting TFRecord file to another specified GCS bucket.

(2)`src/image_train_preparation/pipfile` and `src/image_train_preparation/Pipfile.lock` - These files specify the required packages for building the container.

(3)`src/image_train_preparation/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(4)`src/image_train_preparation/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(5)`src/image_train_preparation/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as tensorizing images and uploading the processed data to the target GCS bucket.

To run Dockerfile - enter the below commands in your local terminal:

-   cd ~/src/image_train_preparation/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

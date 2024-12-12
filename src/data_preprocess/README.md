#### Preprocess container (data_preprocess)

-   This container reads local data (downloaded from Kaggle), resizes the images to 224x224, perform data augmentation (flipping, rotating +/- 15 degrees, adjusting brightness) on the images, generate a CSV file that saves image paths and their corresponding labels, and save all images (original and augmented) to GCS bucket.
-   Input to this container is local image folder location and destination GCS Bucket, secrets needed - via docker
-   Output from this container stored at GCS Bucket

(1)`src/data-preprocess/data_loader.py` - Here we upload our local data to the destination GCS Bucket.

(2)`src/data-preprocess/preprocess.py` - Here we preprocess the Stanford Cars data set. We resize images to 224x224, apply data augmentation (including flipping, rotating by ±15 degrees, and brightness adjustments), rename the augmented images with their methods attached to their file names ("flip", "bright", "dark", "rot-15", "rot15"), creates a CSV file recording image paths and corresponding labels, and uploads both original and augmented images to a GCS bucket.

(3)`src/data-preprocess/Pipfile` and `src/data-preprocess/Pipfile.lock` - These files specify the required packages and their versions for building the container.

(4)`src/data-preprocess/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `Pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(5)`src/data-preprocess/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(6)`src/data-preprocess/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as preprocessing images from the local folder and uploading the processed data to the target GCS bucket.

To run Dockerfile - enter the below commands in your local terminal:

-   cd ~/src/data-preprocess/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

#### Model training container (model_training)

-   This container is a proof of principal that the model training step is containerized, and could be completed by submitting a cloud run job on GCP. Due to time limitation, we will use Google Colab with GPU resources to train the actual model that we deploy for our application.
-   This container includes a script for performing transfer learning on an InceptionV3-based model using a randomly sampled set of 20,000 images from the augmented pool of approximately 90,000 images. As advised by our teaching fellow, Javier, the complete random sampling method poses a high risk of class imbalance in the dataset used for model training, which needs to be addressed through stratified sampling or downsampling in future training iterations.
-   Input to this container is the GCS file location of the tensorized record, the hyperparameters required for model training (e.g. batch size, epoch number, image height, image width, etc.) and the destination GCS Bucket that saves the trained model, secrets needed - via docker
-   Output from this container stored at GCS Bucket

(1)`src/model-training/model_training_v3.py` - This script uses the TFRecord dataset in the GCS Bucket with tensorized TFRecord, preprocesses the data, and trains a deep learning model using transfer learning with InceptionV3.

(2)`src/model-training/pipfile` and `src/model-training/Pipfile.lock` - These files specify the required packages for building the container.

(3)`src/model-training/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(4)`src/model-training/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(5)`src/model-training/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as training the model and uploading the model parameters to the target GCS bucket.

To run Dockerfile - enter the below commands in your local terminal:

-   cd ~/src/model-training/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh
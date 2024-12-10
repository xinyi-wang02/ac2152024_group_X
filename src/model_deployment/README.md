#### Model deployment container (model-deployment)

-   This container uploads our complete model, which was trained using a Colab notebook with GPU resources, to the GCP Bucket and deploys it. The structure of the notebook follows the same format as `src/model-training/model_training_v3.py`.
-   Input to this container is the GCS file location of the saved model and model display name.
-   Output from this container is an endpoint created by Vertex AI for serving predictions from the deployed model.

(1)`src/model-deployment/model_deployment.py` - This script uploads a model from a saved model on GCS to Vertex AI and uses a prebuilt TensorFlow container for inference. It then deploys the model to an endpoint with configurations, including machine type, traffic distribution, and replica counts, and performs the deployment asynchronously.

(2)`src/model-deployment/pipfile` and `src/model-deployment/Pipfile.lock` - These files specify the required packages for building the container.

(3)`src/model-deployment/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(4)`src/model-deployment/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(5)`src/model-deployment/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as training the model and uploading the model parameters to the target GCS bucket.

To run Dockerfile - enter the below commands in your local terminal:

-   cd ~/src/model-deployment/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh
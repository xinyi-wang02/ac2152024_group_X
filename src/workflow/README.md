#### Pipeline container (workflow)

-   This container sets up the execution of our pipeline in Vertex AI, which includes data preprocessing, image tensor preparation, model training, and deployment.
-   Input to this container is our Google Cloud project ID, service account, registered Docker images for each pipeline component, and storage bucket URIs.
-   Output from this container is a compiled pipeline YAML file and a job run submitted to Vertex AI.

(1)`src/workflow/workflow.py` - This script defines a machine learning pipeline using Vertex AI and Kubeflow Pipelines that includes four main container components: data preprocessing, image tensor preparation, model training, and model deployment. It compiles the pipeline into a YAML file and submits it to Vertex AI for execution, using Docker images stored in Google Cloud and running with a service account.

(2)`src/workflow/pipeline.yaml` - This file defines the structure and execution flow of our Vertex AI pipeline. It specifies the four main components of our pipeline and each of them is linked to a specific container image. The deployment section details the Docker images used for each task to ensure reproducibility and consistency across runs.

(3)`src/workflow/pipfile` and `src/workflow/Pipfile.lock` - These files specify the required packages for building the container.

(4)`src/workflow/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(5)`src/workflow/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(6)`src/workflow/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as training the model and uploading the model parameters to the target GCS bucket.

To run Dockerfile - enter the below commands in your local terminal:

-   cd ~/src/workflow/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

The following is a screenshot of our Vertex AI pipeline and the images that have been uploaded to GCP.

![vertex AI pipeline](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/vertex_ai.png)

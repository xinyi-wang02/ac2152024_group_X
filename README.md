# AC215 - Milestone 5 - CarFever

**Team Members** Nuoya Jiang, Harper Wang

**Group Name** Group 16

**Project** In this project, we aim to develop an application that can accurately identify the car model, make, and year from user-uploaded photos.

### Milestone 5

In this milestone, we completed the following tasks:

- Deploying the frontend and API service containers on Google Kubernetes Engine using Ansible Playbook
- Change the predict API endpoint and Add another API endpoint for model training pipeline trigger
- Change the frontend to include user hints and added docker-entrypoint.sh
- Unit tests across all containers & CI/CD through Github Actions
- Containerized `model_training` for future users to train with GCP GPU resource
- Shoot a [video](https://youtu.be/pk9pNsbs8Lk) and wrote a [Medium Post](Placeholder) to document our project

Project Organization

``` bash
├── LICENSE
├── .github/workflows
│   ├── .github/workflows/lint.yml
│   └── .github/workflows/unit_tests.yml
│   └── .github/workflows/ci-cd.yml
├── notebooks
│   ├── eda_notebook.ipynb
│   └── model_testing.ipynb
├── images
│   ├── api.png
│   ├── coverage.png
│   ├── eda_manu.png
│   ├── eda_year.png
│   ├── frontend.png
│   ├── lint.png
│   ├── sol_arch.png
│   ├── tech_arch.png
│   ├── vertex_ai.png
│   └── wnb_ip.png
├── reports
│   └── AC215_project_proposal_group16_updated.pdf
│   └── milestone4.md
├── README.md
└── src
    ├── api_service
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── dictionary.py
    │   ├── label_dictionary.json
    │   ├── car_preprocessed_folder_class_label_dictionary.csv
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    │   ├── server.py
    │   └── README.md
    ├── deployment
    │   ├── nginx-config
    │   │   └── nginx
    │   ├── docker-shell.sh
    │   ├── Dockerfile
    │   ├── docker-entrypoint.sh
    │   ├── deploy-docker-images.yml
    │   ├── deploy-k8s-cluster.yml
    │   ├── inventory.yml
    │   └── README.md
    ├── data_preprocess_mini
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── data_loader.py
    │   ├── preprocess.py
    │   ├── download.py
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    │   └── README.md
    ├── data_preprocess
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── data_loader.py
    │   ├── preprocess.py
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    │   └── README.md
    ├── frontend
    │   ├── assests
    │   │   └── background_image_aigen.jpg
    │   ├── docker-shell.sh
    │   ├── Dockerfile
    │   ├── Dockerfile.dev
    │   ├── index.html
    │   ├── main.js
    │   ├── styles.css
    │   └── README.md
    ├── image_train_preparation
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── tensorizing.py
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    │   └── README.md
    ├── image_train_preparation-20k
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── tensorizing.py
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    │   └── README.md
    ├── model_deployment
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── model_deployment.py
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    │   └── README.md
    ├── model_training
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── model_training_v3.py
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    │   └── README.md
    ├── pwd
    ├── .pre-commit-config.yaml
    ├── tests
    │   ├── resources
    │   │   └── car_test_mini
    │   │   │   └── test
    │   │   │   │   └── Acura Integra Type R 2001
    │   │   │   │   │   └── 00128.jpg
    │   │   │   │   │   └── 00130.jpg
    │   │   │   │   ├── Acura RL Sedan 2012
    │   │   │   │   │   └── 00183.jpg
    │   │   │   │   │   └── 00249.jpg
    │   │   │   ├── train
    │   │   │   │   └── Acura Integra Type R 2001
    │   │   │   │   │   └── 00198.jpg
    │   │   │   │   │   └── 00255.jpg
    │   │   │   │   ├── Acura RL Sedan 2012
    │   │   │   │   │   └── 00670.jpg
    │   │   │   │   │   └── 00691.jpg
    │   │   ├── upload_test_images
    │   │   │   └── test_images
    │   │   │   │   └── 00128.jpg
    │   ├── conftest.py
    │   ├── run_single_test.sh
    │   ├── run_test.sh
    │   ├── test_api_dict.py
    │   ├── test_data_preprocess.py
    │   ├── test_data_preprocess_mini.py
    │   ├── test_image_train_preparation.py
    │   ├── test_image_train_preparation_20k.py
    │   ├── test_model_deploy.py
    │   ├── test_model_training.py
    │   └── README.md
    ├── workflow
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── entrypoint.sh
    │   ├── workflow.py
    │   ├── docker-shell.sh
    │   ├── pipeline.yaml
    │   └── README.md
    ├── Dockerfile
    ├── Pipfile
    ├── Pipfile.lock
    ├── pre-commit-config.yaml
    ├── entrypoint.sh
    └── docker-shell.sh
```

The following are some sample uses of our app:

![01001](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/01001.jpg)
![01001_pred](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/01001_pred.png)
![01007](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/01007.jpg)
![01007_pred](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/01007_pred.png)

#### Kubernetes Deployment

We deployed the frontend and API service containers to the Kubernetes cluster to address key distributed systems challenges, such as load balancing. The pipeline includes a Dockerfile and Ansible playbooks that build and push container images for the frontend and API services to Artifact Registry. It also deploys a Kubernetes cluster on Google Kubernetes Engine (GKE) using Ansible, with support for GKE cluster autoscaling through node pool scaling. The entire pipeline is orchestrated with shell scripts that manage GCP authentication, and configure secrets to fully automate the deployment process.

The following is a screenshot of the Kubernetes cluster we are running in GCP:

![GKE](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/gke.jpg)

#### Deployment Pipeline (`src/deployment/`)

(1)`src/deployment/deploy-docker-images.yml` - This yaml file builds Docker images for frontend and API services and pushes them to Artifact Registry.

(2)`src/deployment/deploy-k8s-cluster.yml` - This yaml file creates and manages a Kubernetes cluster on GKE using Ansible Playbook. It deploys frontend and API services along with Nginx Ingress. It defines and applies Kubernetes manifests (Deployments, Services, Ingress) using Ansible's k8s module while also configuring Secrets and ConfigMaps for the pods.

(3)`src/deployment/inventory.yml` - This yaml file defines hosts and variables for Ansible playbooks. It sets up GCP credentials and SSH information to allow Ansible to manage remote systems.

(4)`src/deployment/Dockerfile` - This file defines the steps for building the containerized environment for deployment on k8s.

(5)`src/deployment/docker-shell.sh` and `src/deployment/docker-entrypoint.sh` - These scripts build and run the Docker container with the context of the Ansible environment. It also specifies credentials for GCP authentication and container entry point.

(6)`src/deployment/nginx-config/nginx/nginx.config` - This file defines the NGINX configuration for routing traffic to the frontend and API services within the Kubernetes cluster. It sets up proxy rules to forward requests to the API service on port 9000 and to the frontend at its root path. 
#### Set up instructions

Prerequisites
- Docker, gcloud CLI
- Get GCP Service Account Key file, and set environment variables (GCP Project ID, region, and zone)
- Create folder `~/src/deployment/secrets/`
- Copy secrets json file to `~/src/deployment/secrets/`

Deployment instructions
In your local terminal, type the following commands:
- cd ~/src/deployment/
- chmod +x docker-shell.sh
- ./docker-shell.sh

Once the container is running, ensure it has access to GCP services by typing the following commands in gcloud CLI:
- gcloud auth configure-docker us-central1-docker.pkg.dev

Build and Push Docker Containers to Artifact Registry by typing the following command:
- ansible-playbook deploy-docker-images.yml -i inventory.yml

After pushing the images, type the following command to deploy the Kubernetes Cluster:
- ansible-playbook src/deployment/deploy-k8s-cluster.yml -i src/deployment/inventory.yml --extra-vars cluster_state=present

Note the nginx_ingress_ip that was outputted by the cluster command
- Visit http:// YOUR INGRESS IP.sslip.io to view the website

The following are screenshots of logs during deployment on Kubernetes.

![k8s_1](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/k8s_1.jpg)
![k8s_2](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/k8s_2.jpg)
![k8s_3](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/k8s_3.jpg)
![k8s_4](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/k8s_4.jpg)
![k8s_5](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/k8s_5.jpg)

#### API container (api_service)

- We have major change from the previous milestone. For our /predict endpoint, it now not only predicts the label for a user uploaded image, but it also uploads the image to our raw data bucket accroding to the predicted label. We also have a new endpoint called /trigger_pipeline. This endpoint deploy our backend ML pipeline on Vertex AI once we have enough new data points (1000 for now). This pipeline is a bit different from our initial vertex AI pipeline: after preprocessing, preparatin, training and validation check, the model is not deployed to a new endpoint. Instead, the new version is pushed the model registry, so we still have the same model serving endpoint.
- We also integrated the API service with an automated deployment powered by **Ansible playbook** and a **Kubernetes cluster**.

To run Dockerfile - follow the steps below:
- create folder `~/src/api-service/no_ship/`
- copy secret json file to `~/src/api-service/no_ship/`

in your local terminal, type the following commands:
- cd ~/src/api-service/
- chmod +x docker-shell.sh
- ./docker-shell.sh
- when testing locally, go to localhost/9000/docs on your browswer to look at the graphic UI
The following is an updated screenshot of our FastAPI `docs` interface.

![image](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/api_new.png)

The original one contains the result from the `/predict` endpoint at the bottom.

![image](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/api.png)

#### Frontend container (frontend)

- We have added a hint to our frontend. We noticed this error: google.api_core.exceptions.FailedPrecondition: 400 The request size (4027510 bytes) exceeds 1.500MB limit. So if the user uploads an image exceeding 1.5 MB, our fronted will hint the user to change to a smaller image. Note that this folder contains docker files for both development and production. We integrated the frontend with an automated deployment powered by **Ansible playbook** and a **Kubernetes cluster**.

In order to run the app on local, we first need to set up and run the API container.

Before running the container for frontend, modify `docker-shell.sh` to change the docker image to `Dockerfile.dev`

Then, we continue in the frontend container, type the following commands:
-   cd ~/src/frontend/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

and paste "127.0.0.1:8080" in your browser to interact with the webpage.

#### CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

We added CI/CD using GitHub Actions, such that we can trigger deployment using GitHub Events. Our yaml file which instantiates the actions can be found in `.github/workflows/ci-cd.yml`.

Our CI/CD pipeline is triggered on pushes to the `main` or `harper_test_2` branches if the commit message contains "/run-ci-cd". It builds and runs a Docker container for deployment, authenticates to Google Cloud, and uses a containerized app to deploy services for the pipeline while ensuring GCP credentials are securely passed to the container.

#### CI/CD Pipeline (`.github/workflows/unit_tests.yml`)

Our unit_test now has a coverage of 82% and it was 58% before. We added many more test functions to ensure our pipeline can run smoothly everytime we would like to retrain and push a new version of model to model registry. We also functionized our training script in the model-training container and our tests covered everything except the signature of the model. 

#### CI/CD Pipeline (`.github/workflows/lint.yml`)

We did not change the lint file. We used ruff as our linting tool. Everytime before we want to commit, we run the command pre-commit run --all-files.

#### Test Container and Documentation (tests)

- This container runs a series of tests to verify data preprocessing, tensorizing, model training, model deployment, and one API service's functionalities. PyTest is used to ensure accurate data upload, processing, tensorization from cloud storage, and to test model training and deployment functionalities as much as possible.
- The coverage report indicates an 82% coverage rate.

Detailed Report on Uncovered Parts

- **`src/data_preprocess/data_loader.py`**  
  Lines 51-61: Argparse arguments.

- **`src/data_preprocess/preprocess.py`**  
  Line 64: Requires >1000 images to cover, but a mini dataset was used for testing.  
  Lines 72-73: Coverage requires handling all edge cases. Attempts to cover this using the function `test_process_images_exception_handling` in `src/tests/test_data_preprocess.py` did not trigger coverage.  
  Lines 77-114: Argparse arguments.

- **`src/data_preprocess_mini/data_loader.py`**  
  Lines 51-61: Argparse arguments.

- **`src/data_preprocess_mini/download.py`**  
  Lines 26-35: Argparse arguments.

- **`src/data_preprocess_mini/preprocess.py`**  
  Line 69: Requires >1000 images to cover, but a mini dataset was used for testing.  
  Lines 77-78: Coverage requires handling all edge cases. Attempts to cover this using the function `test_process_images_exception_handling` in `src/tests/test_data_preprocess_mini.py` did not trigger coverage.  
  Lines 82-119: Argparse arguments.

- **`src/image_train_preparation/tensorizing.py`**  
  Lines 96-140: Argparse arguments.

- **`src/image_train_preparation_20k/tensorizing.py`**  
  Line 45: Requires >100 images to cover, but a mini dataset was used for testing.  
  Lines 101-159: Argparse arguments.

- **`src/model_training/model_training_v3.py`**  
  Lines 87-113: Coverage requires outputs from other functions, which are difficult to simulate in the test environment.  
  Lines 118-175: Argparse arguments.  
  Lines 179-217: Argparse arguments.

- **Test Scripts**  
  Files such as `conftest.py`, `test_api_dict.py`, ... , `test_model_deploy.py`, and `test_model_training.py` themselves are not covered in the report.

### Untested Scripts

- **`src/api_service/dictionary.py`**  
  Tested with `test_api_dict.py` but not included in the coverage report because the tested script is not functionalized and is not imported into the test script.

- **`src/api_service/server.py`**  
  Not required to test.

- **`src/deployment/`**  
  No `.py` files to test.

- **`src/frontend/`**  
  No `.py` files to test.

- **`src/model_deployment/`**  
  Tested with `test_model_deploy.py` but not included in the coverage report because the tested script is not functionalized and is not imported into the test script.

- **`src/workflow/`**  
  Not tested as the script's logic is simple (running all Docker images for model deployment on Vertex AI), and a similar procedure is repeated using Ansible in `src/deployment/`.

Instructions to Run Tests Manually

enter the below commands in your local terminal:
-   cd ~/src/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

if the user would like to test a single function, type the following after container is running:
-   ./run_single_test.sh

if the user would like to test all function, type the following after container is running:
-   ./run_test.sh

The user could change the function that they would like to test in `run_single_test.sh`

The following is a screenshot of our coverage report.

![coverage_new](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/coverage_new.png)

The following is a screenshot of our linting test.

![lint](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/lint.png)

#### Code Structure

The following are the folders in `/src/` and their modified changes since milestone 4:

-   api_service: added an endpoint to collect user-uploaded images and trigger the pipeline, which includes **data preprocessing, tensorization, and model training** to retrain the model
-   data_preprocess
-   data_preprocess_mini
-   deployment: deployed the frontend and API service containers on Kubernetes using Ansible
-   frontend: added another API endpoint for model training pipeline trigger
-   image_train_preparation
-   image_train_preparation_20k
-   model_deployment
-   model_training: containerized for users with GCP GPU resources for future training
-   tests: more test scripts to cover 82% of the lines
-   workflow

#### Limitations and Notes

**Limitations:**

1. **Image Resizing Impact**  
   Based on feedback from the last milestone, we visualized the original and resized images to determine if resizing images would alter the proportions of the car images used for training. Our analysis confirmed that resizing does, in fact, change the car's proportions, which may negatively affect model accuracy. Future work could explore training with different image sizes or maintaining the original dimensions to evaluate whether this approach improves model performance.

   ![resize_224](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/resize_224.png)

2. **Class Imbalance**  
   Due to time constraints, we were unable to retrain our model using stratified sampled images. As a result, the model's performance may decline when predicting car images from underrepresented classes. Future work could involve web-scraping additional car images to address the class imbalance and enhance overall model performance.

3. **Incorrect Labels**  
   The original dataset contains some incorrect labels, which may negatively impact the model's predictive accuracy. Future work could involve manual labeling to correct these inaccuracies and further improve model performance.

4. **Evaluation Metrics**  
   We acknowledge that performance evaluation should go beyond accuracy. Future work should incorporate additional performance metrics such as precision, recall, and F1 score to provide a more comprehensive assessment of model performance.


---

**Notes:**

For Milestone 4, we constructed two distinct workflows: one for testing and another for model training on Weights & Biases (W&B). We chose to keep both workflows to enable future work on mini-batch data testing. The following outlines the folder structure for different workflows and their shared components:

- **Mini-batch Data Workflow Folders:**  
  - `data_preprocess_mini`  
  - `image_train_preparation`  
  - `model_training`  

- **Shared Folders (for both mini-batch and full model workflows):**  
  - `model_deployment`  
  - `workflow`  
  - `tests`  
  - `api_service`  
  - `frontend`  
  - `deployment`  

- **Full Batch Data Workflow Folders:**  
  - `data_preprocess`  
  - `image_train_preparation_20k`  

Each folder includes a `README.md` file containing detailed instructions on file structures, function descriptions, and replication guides to facilitate seamless navigation and usage for future development.

#### Notebook

As we can see from the following 2 plots from our EDA notebook:

![eda_manu](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/eda_manu.png)
![eda_manu](https://github.com/xinyi-wang02/ac2152024_group_X/blob/main/images/eda_year.png)

There is a noticeable class imbalance in the dataset, with an overrepresentation of Chevrolet cars and cars produced in 2012. To address this, we used stratified sampling before model training to ensure that the training set maintains the same class proportions as the original dataset, which is crucial for handling uneven class distributions. Stratified sampling also prevents the extreme scenario where all sampled images belong to a single class, leading to a model that can only make a limited range of predictions.

In `model_testing.ipynb`, we experimented with 3 different model structure to perform model training (finetuning).

CarNetV1 and CarNetV2 use ResNet152V2, a deep convolutional neural network known for its  effective feature extraction and improved training stability. CarNetV1 uses a single dense layer before the output, while CarNetV2 incorporates two dense layers for potentially more complex feature transformation.

CarNetV3 utilizes InceptionV3, which employs a different architecture focusing on multi-scale feature extraction through its inception modules, providing diversity in feature learning. These variations help us assess which architecture and layer configuration work best for distinguishing between car models, considering both depth and structure differences.

Additionally, we implemented early stopping in all models to monitor validation loss during training and prevent overfitting by stopping training when performance stops improving, which ensures that the models could generalize better to unseen data.


**GCP Bucket Structure**

------------
     multiclass-car-project-demo
      ├── 215-multiclass-car-bucket/
            ├── car_preprocessed_folder/
                  ├── all_images/
                  ├── class_label.csv
                  ├── class_label_dictionary.csv
      ├── 215_deployment/
      ├── car_class_test_bucket/
            ├── test_images/
      ├── mini-215-multiclass-car-bucket/
            ├── car_folder/
                  ├── test/
                  ├── train/
            ├── car_preprocessed_folder/
                  ├── all_images/
                  ├── class_label.csv
                  ├── class_label_dictionary.csv
      ├── mini-pipeline/
            ├── car_preprocessed_folder/
            ├── vertext_pipeline_root/
      ├── mini-tensor-bucket/
            ├── data.tfrecord
      ├── mini_model_wnb/
            ├── assets/
            ├── fingerprint.pb
            ├── saved_model.pb
            ├── variables/
                  ├── variables.data-00000-of-00001
                  ├── variables.index
      ├── model_wnb/
            ├── carnet_v1_50epoch/
                  ├── fingerprint.pb
                  ├── saved_model.pb
                  ├── variables/
                        ├── variables.data-00000-of-00001
                        ├── variables.index
            ├── carnet_v3_4epoch_tf213/
                  ├── fingerprint.pb
                  ├── saved_model.pb
                  ├── variables/
                        ├── variables.data-00000-of-00001
                        ├── variables.index
            ├── carnet_v3_50epoch_tf213/
                  ├── fingerprint.pb
                  ├── saved_model.pb
                  ├── variables/
                        ├── variables.data-00000-of-00001
                        ├── variables.index test_bucket_new
      ├── test_bucket_new
      └── tensor-bucket-20k/
            ├── data.tfrecord

--------

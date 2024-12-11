# AC215 - Milestone 5 - CarFever

**Team Members** Nuoya Jiang, Harper Wang

**Group Name** Group 16

**Project** In this project, we aim to develop an application that can accurately identify the car model, make, and year from user-uploaded photos.

### Milestone 5

In this milestone, we completed the following tasks:

-   Deploying the frontend and API service containers on Kubernetes using Ansible
-   Add another API endpoint for model training pipeline trigger
-   CI/CD through Github Actions
-


Project Organization

``` bash
├── LICENSE
├── .github/workflows
│   ├── .github/workflows/lint.yml
│   └── .github/workflows/unit_tests.yml
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
├── README.md
└── src
    ├── api-service
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── dictionary.py
    │   ├── label_dictionary.json
    │   ├── car_preprocessed_folder_class_label_dictionary.csv
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    │   └── server.py
    ├── data_preprocess_mini
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── data_loader.py
    │   ├── preprocess.py
    │   ├── download.py
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    ├── data-preprocess
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── data_loader.py
    │   ├── preprocess.py
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    ├── frontend
    │   ├── docker-shell.sh
    │   ├── Dockerfile
    │   ├── index.html
    │   ├── main.js
    │   ├── styles.css
    ├── image_train_preparation
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── tensorizing.py
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    ├── image-train-preparation-20k
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── tensorizing.py
    │   ├── entrypoint.sh
    │   ├── docker-shell.sh
    ├── model-deployment
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── model_deployment.py
    │   ├── entrypoint.sh
    │   └── docker-shell.sh
    ├── model-training
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── model_training.py
    │   ├── model_training_inceptionV3.py
    │   ├── best_model.keras
    │   ├── entrypoint.sh
    │   └── docker-shell.sh
    ├── pwd
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
    │   ├── test_data_preprocess_mini.py
    │   ├── test_image_train_preparation.py
    ├── workflow
    │   ├── Dockerfile
    │   ├── pipfile
    │   ├── Pipfile.lock
    │   ├── entrypoint.sh
    │   ├── workflow.py
    │   ├── docker-shell.sh
    │   └── pipeline.yaml
    ├── Dockerfile
    ├── Pipfile
    ├── Pipfile.lock
    ├── pre-commit-config.yaml
    ├── entrypoint.sh
    └── docker-shell.sh
```

The following are some sample uses of our app:

![Image](placeholder)
![Image](placeholder)
![Image](placeholder)
![Image](placeholder)

#### Kubernetes Deployment

We deployed the frontend and API service containers to the Kubernetes cluster to address key distributed systems challenges, such as load balancing. The pipeline includes a Dockerfile and Ansible playbooks that build and push container images for the frontend and API services to Google Container Registry (GCR). It also provisions a Kubernetes cluster on Google Kubernetes Engine (GKE) using Ansible, with support for GKE cluster autoscaling through node pool scaling. The entire pipeline is orchestrated with shell scripts that manage GCP authentication, configure secrets, and implement infrastructure as code (IaC) to fully automate the deployment process.

The following is a screenshot of the Kubernetes cluster we are running in GCP:

![Image](placeholder)

#### Deployment Pipeline (`src/deployment/`)

(1)`src/deployment/deploy-docker-images.yml` - This yaml file builds Docker images for frontend and API services and pushes them to GCR.

(2)`src/deployment/deploy-k8s-cluster.yml` - This yaml file creates and manages a Kubernetes cluster on GCP using Ansible and GCP Cloud modules. It deploys frontend and API services along with Nginx Ingress, Persistent Volumes, and Secrets. It defines and applies Kubernetes manifests (Deployments, Services, Ingress) using Ansible's k8s module while also configuring Secrets and ConfigMaps for the pods.

(3)`src/deployment/inventory.yml` - This yaml file defines hosts and variables for Ansible playbooks. It sets up GCP credentials and SSH information to allow Ansible to manage remote systems.

(4)`src/deployment/Dockerfile` - This file defines the steps for building the containerized environment.

(5)`src/deployment/docker-shell.sh` and `src/deployment/docker-entrypoint.sh` - These scripts build and run the Docker container with the context of the Ansible environment. It also specifies credentials for GCP authentication and container entry point.

(6)`src/deployment/nginx-config/nginx/nginx.config` - This file defines the NGINX configuration for routing traffic to the frontend and API services within the Kubernetes cluster. It sets up proxy rules to forward requests to the API service on port 9000 and to the frontend at its root path. The configuration also includes server directives for access logging, gzip compression, SSL protocols, and connection settings to support load balancing, request forwarding, and optimized performance.

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
- gcloud auth activate-service-account --key-file /secrets/deployment.json
- gcloud config set project $GCP_PROJECT
- gcloud auth configure-docker us-central1-docker.pkg.dev

Build and Push Docker Containers to GCR (Google Container Registry) by typing the following command:
- ansible-playbook deploy-docker-images.yml -i inventory.yml

After inside the container, type the following command to deploy the Kubernetes Cluster:
- ansible-playbook src/deployment/deploy-k8s-cluster.yml -i src/deployment/inventory.yml --extra-vars cluster_state=present

Note the nginx_ingress_ip that was outputted by the cluster command
- Visit http:// YOUR INGRESS IP.sslip.io to view the website

If you want to run our ML pipeline, checkout the directions under `src/workflow/`. The following are the commands to run the pipelines:

- cd ~/src/workflow/
- chmod +x docker-shell.sh
- ./docker-shell.sh

The following are screenshots of logs during deployment on Kubernetes.

![image](placeholder)

#### API container (api_service)

- Most of the container's content remains unchanged from the previous milestone. However, we added an endpoint to collect user-uploaded images and trigger the pipeline, which includes **data preprocessing, tensorization, and model training** to retrain the model.
- Instead of deploying the retrained model immediately, we implemented a validation check to ensure that only models with good performance are deployed, preventing interference with the existing `/predict` endpoint.
- We also integrated the API service with an automated deployment powered by **Ansible playbook** and a **Kubernetes cluster**.

To run Dockerfile - follow the steps below:
- create folder `~/src/api-service/no_ship/`
- copy secret json file to `~/src/api-service/no_ship/`

in your local terminal, type the following commands:
- cd ~/src/api-service/
- chmod +x docker-shell.sh
- ./docker-shell.sh

The following is an updated screenshot of our FastAPI `docs` interface.

![image](placeholder)

#### Frontend container (frontend)

- Most of the container's content remains unchanged from the previous milestone. Note that this folder contains docker files for both development and production. We integrated the frontend with an automated deployment powered by **Ansible playbook** and a **Kubernetes cluster**.

In order to run the app on local, we first follow the steps below to set up and run the API container:
-  create folder `~/src/api-service/no_ship/`
-  copy secret json file

in your local terminal, type the following commands:
-   cd ~/src/api-service/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

Before running the container for frontend, modify `docker-shell.sh` to change the docker image to `Dockerfile.dev`

Then, we continue in the frontend container, type the following commands:
-   cd ~/src/frontend/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

After the container is running, type the following command:
-   http-server

and paste "127.0.0.1:8080" in your browser to interact with the webpage.

#### CI/CD Pipeline (`.github/workflows/`)

#### Test container and Documentation (tests)

-   This container runs a series of tests to verify data preprocessing, image processing, and data download functionalities, using PyTest to ensure that data is accurately uploaded, processed, and downloaded from cloud storage.
-   Input to this container are test scripts, resource images, and configuration files.
-   Output from this container are test outcomes and coverage reports.

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

![coverage report](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/coverage.png)

The following is a screenshot of our linting test.

![lint](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/lint.png)

#### Code Structure

The following are the folders in `/src/` and their modified changes since milestone 4:

-   api_service:
-   data_preprocess
-   data_preprocess_mini
-   deployment:
-   frontend:
-   image_train_preparation
-   image_train_preparation_20k
-   model_deployment
-   model_training
-   tests:
-   workflow

#### Limitations and Notes

NOTE:
There are 2 different workflows under construction now. The milestone 4 deliverable is completed with a mini batch of our data to demonstrate our Vertex AI pipeline (data preprocessing, data tensorizing, model training, and model deployment), API service, frontend, CI/CD, and testing.

For our final workflow, we will perform data preprocessing, data tensorizing, and model deployment using model trained with Google Colab and saved on Weights and Biases. This is currently under develop with scripts in the folder `src/image-train-preparation-20k` that performs stratified sampling from the ~90,000 augmented car images, and trained with Google Colab notebook formated as `src/model-training/model_training_inceptionV3.py`.


#### Notebook

As we can see from the following 2 plots from our EDA notebook:

![eda_manu](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/eda_manu.png)
![eda_manu](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/eda_year.png)

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
      └── tensor-bucket-20k/
            ├── data.tfrecord

--------

# AC215 - Milestone 5 - CarFever

**Team Members** Nuoya Jiang, Harper Wang

**Group Name** Group 16

**Project** In this project, we aim to develop an application that can accurately identify the car model, make, and year from user-uploaded photos.

### Milestone 5

In this milestone, we completed the following tasks:

-   Deploying on Kubernetes using Ansible
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

#### Kubernetes Deployment



The following is a screenshot of the Kubernetes cluster we are running in GCP:

![Image](placeholder)

NOTE:
There are 2 different workflows under construction now. The milestone 4 deliverable is completed with a mini batch of our data to demonstrate our Vertex AI pipeline (data preprocessing, data tensorizing, model training, and model deployment), API service, frontend, CI/CD, and testing.

For our final workflow, we will perform data preprocessing, data tensorizing, and model deployment using model trained with Google Colab and saved on Weights and Biases. This is currently under develop with scripts in the folder `src/image-train-preparation-20k` that performs stratified sampling from the ~90,000 augmented car images, and trained with Google Colab notebook formated as `src/model-training/model_training_inceptionV3.py`.


#### Dataset

Kaggle Stanford Cars - [link](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder/data)


#### API container (api-service)

-   This container hosts a FastAPI server that handles image uploads, processes them, sends the data to a deployed Vertex AI model for car type predictions, and maps the model's output to human-readable car model names.
-   Input to this container is an HTTP POST request to the `/predict` endpoint with an image file in the request body.
-   Output from this container is a JSON response containing the predicted car types based on the image, structured as `{"predicted_car_types": [predicted_labels]}`.

(1)`src/api-service/server.py` - This script defines a FastAPI-based web server that handles HTTP POST requests at the `/predict` endpoint to receive user-uploaded images, process them, and return a prediction of car types. It also includes a function to send the image data to a deployed machine learning model hosted on Google Cloud's Vertex AI for inference.

(2)`src/api-service/label_dictionary.json` - This file is generated by `src/api-service/dictionary.py` and contains a mapping of car model names to their corresponding encoded numerical labels. It is used in `src/api-service/server.py` to translate predicted class numbers from the model into human-readable car model names for user-friendly outputs.

(3)`src/api-service/car_preprocessed_folder_class_label_dictionary.csv` - This file is produced by the `data-preprocess` container and serves as input to `src/api-service/dictionary.py` for extracting the relationship between encoded class numbers and actual car model names.

(4)`src/api-service/dictionary.py` - This script processes `src/api-service/car_preprocessed_folder_class_label_dictionary.csv` to create a dictionary that is subsequently used by `src/api-service/server.py` for mapping predicted class numbers to car model names.

(5)`src/api-service/pipfile` and `src/api-service/Pipfile.lock` - These files specify the required packages for building the container.

(6)`src/api-service/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(7)`src/api-service/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(8)`src/api-service/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as training the model and uploading the model parameters to the target GCS bucket.

To run Dockerfile - follow the steps below:
-   create folder `~/src/api-service/no_ship/`
-   copy secret json file to `~/src/api-service/no_ship/`

in your local terminal, type the following commands:
-   cd ~/src/api-service/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

The following is a screenshot of our FastAPI `docs` interface.

![API screenshot](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/api.png)


#### Frontend container (frontend)

-   This container sets up a web-based frontend application that allows users to upload car images, sends them to an API for model prediction, and displays the results.
-   Input to this container is user-uploaded car images.
-   Output from this container is the predicted car model name, make, and year.

(1)`src/frontend/main.js` - This script enables a form submission event listener that handles user-uploaded car images, sends them to the API endpoint for prediction, and displays the result or an error message on the web page.

(2)`src/frontend/index.html` - This HTML script sets up the web page that allows users to upload a car photo and displays the prediction result after interacting with the backend API through JavaScript..

(3)`src/frontend/styles.css` - This CSS script styles the webpage by defining general layout properties, creating a container with a form and result box, and adding background image.

(4)`src/workflow/Dockerfile` - This file defines the steps for building the container.

(5)`src/workflow/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(6)`src/frontend/assets/background_image_aigen.jpg` - This image serves as the background for the application. It was generated using the latest DALL·E model from OpenAI with the prompt: "I am developing a website for car model identification where users can upload a photo of a car, and I will provide a prediction of the car model. I need a background image that displays some cars with a level of opacity so that it complements rather than overwhelms the content and functionality."

##### Application components

Header: A static header displaying the name of the application.

Interactive window: This section features a white box designed for user interaction. Click the "Choose File" button to upload a car image, and then select "Upload and Identify" to receive the result shortly after.

##### Setup instructions

In order to run the app on local, we first follow the steps below to set up and run the API container:

-   create folder `~/src/api-service/no_ship/`
-   copy secret json file to `~/src/api-service/no_ship/`

in your local terminal, type the following commands:
-   cd ~/src/api-service/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

Then, we continue in the frontend container, type the following commands:
-   cd ~/src/frontend/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

After the container is running, type the following command:
-   http-server

and paste "127.0.0.1:8080" in your browser to interact with the webpage.

##### Usage guidelines

After setting up the application as described above, you can upload car images and receive predictions from our model. Please ensure that the images are static and in formats such as `.jpg`, `.jpeg`, `.png`, `.webp`, etc.

The following is a screenshot of our frontend with an example.

![frontend example](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/frontend.png)

#### Test container and Documentation (tests)

-   This container runs a series of tests to verify data preprocessing, image processing, and data download functionalities, using PyTest to ensure that data is accurately uploaded, processed, and downloaded from cloud storage.
-   Input to this container are test scripts, resource images, and configuration files.
-   Output from this container are test outcomes and coverage reports.

(1)`src/tests/test_data_preprocess_mini.py` - The script tests three main functions related to data preprocessing.

-   The `test_uploader` function verifies that images can be successfully uploaded to the GCS bucket with the `upload_directory_with_transfer_manager` function in `src/data_preprocess_mini/preprocess.py`.
-   The `test_process_images` function confirms that the function `process_images` in `src/data_preprocess_mini/preprocess.py` processes and outputs images correctly.
-   The `test_download` function checks if images can be accurately downloaded from the storage bucket with the `download` function in `src/data_preprocess_mini/download.py`.

(2)`src/tests/test_image_train_preparation.py` - The script tests the `download_data` function in `src/image_train_preparation/tensorizing.py` to ensure it can accurately download image and CSV data from a specified cloud storage bucket and save it to a local directory, verifying that the expected number of images and CSV files are present after the download.

(3)`src/tests/conftest.py` - This is a conventional script that we include in testing.

(4)`src/tests/run_single_test.sh` - This script runs the test function `test_download_data`, from `test_image_train_preparation.py` to verify that the `download_data` function in `src/image_train_preparation/tensorizing.py`.

(5)`src/tests/run_test.sh` - This script runs all test functions within `src/tests` using pytest while collecting coverage data for the code in the parent directory (excluding the files in `src/tests`), and generates the coverage report to identify which lines of code were executed during testing and which were missed.

(6)`src/Pipfile` and `src/Pipfile.lock` - These files specify the required packages for building the container.

(7)`src/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `Pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(8)`src/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(9)`src/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as training the model and uploading the model parameters to the target GCS bucket.

(10)`src/tests/resources` - This folder contains the images that were used during testing.

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


#### How to check CI/CD test results?

The functions that we are testing are listed in the section above. We tested the function to upload and download files between local and GCS Bucket because we want to ensure that our file transfer workflow is smooth and errorless. We also tested the functions that preprocesses images because we need to ensure that the preprocessing procedures are consistent and replicable for all processed images.

In order to check our CI/CD results, go to `Actions` tab of this repo, find `Pipeline Unit Tests` in the panel on the left, you can see a log of automated testing for each push.

We will continue to test more functions and integrate our entire pipeline with the existing test workflow.

#### Why linting and how to perform linting?

Linting is essential in this project because it helps maintain code quality, readability, and consistency. By automatically checking for programming errors, potential bugs, and stylistic issues, linting ensures that the code adheres to best practices and coding standards.

We used [Ruff](https://docs.astral.sh/ruff/) and [pre-commit](https://pre-commit.com/) to implement our linting pipeline. We specified our configuration in `.pre-commit-config.yaml` to remove any trailing whitespaces at the end of lines in the code files and ensures that all files end with a newline. Then, we add `.github/workflows/lint.yml` so that it automatically runs pre-commit linting checks whenever code is pushed to the main or `milestone4` branches or when a pull request is created, ensuring that any code changes comply with the project's linting rules before merging.

In order to check our linting results, go to `Actions` tab of this repo, find `Pipeline Lint` in the panel on the left, you can see a log of automated linting check for each push.


#### Notebook

As we can see from the following 2 plots from our EDA notebook:

![eda_manu](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/eda_manu.png)
![eda_manu](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/eda_year.png)

There is a noticeable class imbalance in the dataset, with an overrepresentation of Chevrolet cars and cars produced in 2012. To address this, we used stratified sampling before model training to ensure that the training set maintains the same class proportions as the original dataset, which is crucial for handling uneven class distributions. Stratified sampling also prevents the extreme scenario where all sampled images belong to a single class, leading to a model that can only make a limited range of predictions.

In `model_testing.ipynb`, we experimented with 3 different model structure to perform model training (finetuning).

CarNetV1 and CarNetV2 use ResNet152V2, a deep convolutional neural network known for its  effective feature extraction and improved training stability. CarNetV1 uses a single dense layer before the output, while CarNetV2 incorporates two dense layers for potentially more complex feature transformation.

CarNetV3 utilizes InceptionV3, which employs a different architecture focusing on multi-scale feature extraction through its inception modules, providing diversity in feature learning. These variations help us assess which architecture and layer configuration work best for distinguishing between car models, considering both depth and structure differences.

Additionally, we implemented early stopping in all models to monitor validation loss during training and prevent overfitting by stopping training when performance stops improving, which ensures that the models could generalize better to unseen data.

Here is an example screenshot showing the metrics from our model experiments logged in Weights and Biases.

![wnb_ip](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/wnb_ip.png)

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

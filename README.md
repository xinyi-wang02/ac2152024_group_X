# Team CarFever Milestone 4 Deliverable

AC215 - Milestone 4

Project Organization

``` bash
├── LICENSE
├── notebooks
│   ├── eda_notebook.ipynb
│   └── model_testing.ipynb
├── images
│   ├── 
│   └── 
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
    ├── data_preprocess
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
    ├── 
    ├── entrypoint.sh
    └── docker-shell.sh
```

# AC215 - Milestone 4 - CarFever

**Team Members** Nuoya Jiang, Harper Wang

**Group Name** Group 16

**Project** In this project, we aim to develop an application that can accurately identify the car model, make, and year from user-uploaded photos.

### Milestone 4

In this milestone we re-constructed most parts of the project in a new repository:

```         
  (1) Preprocess container [/src/data-preprocess/](src/data-preprocess/)
  
  (2) Tensorizing container [/src/image_train_preparation/](src/image_train_preparation/)
  
  (3) Model training container [/src/model-training/](src/model-training/)

  (4) Model deployment on VertexAI container [/src/model-deployment/](src/model-deployment/)

  (5) Model pipeline [/src/workflow/](src/workflow/)
  
  (6) API service [/src/api-service/](src/api-service/)
   
  (7) Frontend Simple container [/src/frontend/](src/frontend/)
  
  (8) Test container and documentation [/src/tests/](src/tests/)
  
  (9) Notebook explanation
  
  (10) GCP Bucket structure
```

#### Application Design

Below are the Solutions Architecture and Technical Architecture diagrams. These diagrams illustrate how the different system components interact to classify car images.

**Solution Architecture** ![image](Placeholder)

**Technical Architecture** ![image](Placeholder)

#### Dataset

Kaggle Stanford Cars - [link](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder/data)

#### Preprocess container (data-preprocess)

-   This container reads local data (downloaded from Kaggle), resizes the images to 224x224, perform data augmentation (flipping, rotating +/- 15 degrees, adjusting brightness) on the images, generate a CSV file that saves image paths and their corresponding labels, and save all images (original and augmented) to GCS bucket.
-   Input to this container is local image folder location and destincation GCS Bucket, secrets needed - via docker
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

#### Mini Preprocess container (data_preprocess_mini)

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

#### Tensorizing container (image-train-preparation-20k)

#### Model training container (model-training)

-   This container is a proof of principal that uses a mini batch of the original data to perform transfer learning using the baseline model (ResNet50). Due to time limitation, we will use Google Colab with GPU resources to train the actual model that we deploy for our application.
-   This container includes a script for performing transfer learning on an InceptionV3-based model using a randomly sampled set of 20,000 images from the augmented pool of approximately 90,000 images. As advised by our teaching fellow, Javier, the complete random sampling method poses a high risk of class imbalance in the dataset used for model training, which needs to be addressed through stratified sampling or downsampling in future training iterations.
-   Input to this container is the GCS file location of the tensorized record, the hyperparameters required for model training (e.g. batch size, epoch number, image height, image width, etc.) and the destination GCS Bucket that saves the trained model, secrets needed - via docker
-   Output from this container stored at GCS Bucket

(1)`src/model-training/model_training.py` - This script downloads a TFRecord dataset from the GCS Bucket with tensorized TFRecord, preprocesses the data, and trains a deep learning model using transfer learning with ResNet50. After training, it evaluates the model and saves it to GCS with an appropriate serving function for API deployment.

(2)`src/model-training/model_training_inceptionV3.py` - This script downloads a TFRecord dataset from the GCS Bucket with tensorized TFRecord, preprocesses the data, and trains a deep learning model using transfer learning with InceptionV3.

(3)`src/model-training/pipfile` and `src/image_train_preparation/Pipfile.lock` - These files specify the required packages for building the container.

(4)`src/image_train_preparation/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(5)`src/image_train_preparation/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(6)`src/image_train_preparation/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as training the model and uploading the model parameters to the target GCS bucket.

To run Dockerfile - enter the below commands in your local terminal:

-   cd ~/src/model-training/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

#### Model deployment container (model-deployment)

-   This container uploads the mini-model that was trained with `src/model-training/model_training.py` to the GCP Bucket and deploy it.
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

The following is a screenshot of our Vertex AI pipeline and the images that have been uploaded to GCP. ![image](<img width="748" alt="vertex AI pipeline" src="https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/vertex_ai.png">) In the final version of the pipeline, model training will be excluded as we plan to train our final model using Google Colab for easier access to GPU resources. However, this demonstration showcases the full pipeline flow using a smaller model.

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
*create folder `~/src/api-service/no_ship/` 
*copy secret json file to `~/src/api-service/no_ship/`

in your local terminal, type the following commands: 
* cd \~/src/api-service/ 
* chmod +x docker-shell.sh 
* ./docker-shell.sh

The following is a screenshot of our FastAPI `docs` interface. ![image](<img width="748" alt="API screenshot" src="https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/api.png">)

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
* cd ~/src/api-service/ 
* chmod +x docker-shell.sh 
* ./docker-shell.sh

Then, we continue in the frontend container, type the following commands: 
* cd ~/src/frontend/ 
* chmod +x docker-shell.sh 
* ./docker-shell.sh

After the container is running, type the following command: 
* http-server

and paste "127.0.0.1:8080" in your browser to interact with the webpage.

##### Usage guidelines

After setting up the application as described above, you can upload car images and receive predictions from our model. Please ensure that the images are static and in formats such as `.jpg`, `.jpeg`, `.png`, `.webp`, etc.

The following is a screenshot of our frontend with an example. 
![image](<img width="748" alt="frontend example" src="https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/frontend.png">)

#### Test container and Documentation (tests)

-   This container runs a series of tests to verify data preprocessing, image processing, and data download functionalities, using PyTest to ensure that data is accurately uploaded, processed, and downloaded from cloud storage.
-   Input to this container are test scripts, resource images, and configuration files.
-   Output from this container are test outcomes and coverage reports.

(1)`src/tests/test_data_preprocess_mini.py` - The script tests three main functions related to data preprocessing.

-   The `test_uploader` function verifies that images can be successfully uploaded to the GCS bucket with the `upload_directory_with_transfer_manager` function in `src/data_preprocess_mini/preprocess.py`.
-   The `test_process_images` function confirms that the function `process_images` in `src/data_preprocess_mini/preprocess.py` processes and outputs images correctly.
-   The `test_download` function checks if images can be accurately downloaded from the storage bucket with the `download` function in `src/data_preprocess_mini/download.py`.

(2)`src/tests/test_image_train_preparation.py` - The script tests the `download_data` function in `src/image_train_preparation/tensorizing.py` to ensure it can accurately download image and CSV data from a specified cloud storage bucket and save it to a local directory, verifying that the expected number of images and CSV files are present after the download.

(3)`src/tests/conftest.py` - This script tests functions for managing and counting files in a directory, including removing .DS_Store files and recursively counting files of a specified type?

(4)`src/tests/run_single_test.sh` - This script runs the test function `test_download_data`, from `test_image_train_preparation.py` to verify that the `download_data` function in `src/image_train_preparation/tensorizing.py`.

(5)`src/tests/run_test.sh` - This script runs all test functions within `src/tests` using pytest while collecting coverage data for the code in the parent directory (excluding the files in `src/tests`), and generates the coverage report to identify which lines of code were executed during testing and which were missed.

(6)`src/Pipfile` and `src/Pipfile.lock` - These files specify the required packages for building the container.

(7)`src/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `Pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(8)`src/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(9)`src/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as training the model and uploading the model parameters to the target GCS bucket.

(10)`src/tests/resources` - This folder contains the images that were used during testing.

Instructions to Run Tests Manually:

The following is a screenshot of our coverage report. 
![image](<img width="748" src="https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/coverage.png">)

We will continue to test more functions and integrate our entire pipeline with the existing test workflow.

#### Notebook

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
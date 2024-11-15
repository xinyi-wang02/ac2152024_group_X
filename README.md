Team CarFever Milestone 4 Deliverable
==============================

AC215 - Milestone 4

Project Organization
```bash
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
    ├── docker-compose.yml
    ├── entrypoint.sh
    └── docker-shell.sh
```

# AC215 - Milestone 4 - CarFever

**Team Members**
Nuoya Jiang, Harper Wang

**Group Name**
Group 16

**Project**
In this project, we aim to develop an application that can accurately identify the car model, make, and year from user-uploaded photos. 

### Milestone 4 ###

In this milestone we re-constructed most parts of the project in a new repository:

      (1) Preprocess container [/src/data-preprocess/](src/data-preprocess/)
      
      (2) Tensorizing container [/src/image_train_preparation/](src/image_train_preparation/)
      
      (3) Model training container [/src/model-training/](src/model-training/)
    
      (4) Model deployment on VertexAI container [/src/model-deployment/](src/model-deployment/)

      (5) API service [/src/api-service/](src/api-service/)
      
      (6) Frontend Simple container [/src/frontend/](src/frontend/)
       
      (7)
      
      (8)
      
      (9)
       
      (10)

#### Application Design ####

Below are the Solutions Architecture and Technical Architecture diagrams. These diagrams illustrate how the different system components interact to classify car images.

**Solution Architecture**
![image](<Placeholder>)

**Technical Architecture**
![image](<Placeholder>)

#### Dataset ####

Kaggle Stanford Cars - [link](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder/data)

#### Preprocess container ####

* This container reads local data (downloaded from Kaggle), resizes the images to 224x224, perform data augmentation (flipping, rotating +/- 15 degrees, adjusting brightness) on the images, generate a CSV file that saves image paths and their corresponding labels, and save all images (original and augmented) to GCS bucket.
* Input to this container is local image folder location and destincation GCS Bucket, secrets needed - via docker
* Output from this container stored at GCS Bucket

(1)`src/data-preprocess/data_loader.py` - Here we upload our local data to the destination GCS Bucket.

(2)`src/data-preprocess/preprocess.py` - Here we preprocess the Stanford Cars data set. We resize images to 224x224, apply data augmentation (including flipping, rotating by ±15 degrees, and brightness adjustments), rename the augmented images with their methods attached to their file names ("flip", "bright", "dark", "rot-15", "rot15"), creates a CSV file recording image paths and corresponding labels, and uploads both original and augmented images to a GCS bucket.

(3)`src/data-preprocess/Pipfile` and `src/data-preprocess/Pipfile.lock` - These files specify the required packages and their versions for building the container.

(4)`src/data-preprocess/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `Pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(5)`src/data-preprocess/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(6)`src/data-preprocess/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as preprocessing images from the local folder and uploading the processed data to the target GCS bucket.

To run Dockerfile - enter the below commands in your local terminal:

* cd ~/src/data-preprocess/
* chmod +x docker-shell.sh
* ./docker-shell.sh


#### Tensorizing container (image_train_preparation) ####

* This container downloads preprocessed images and their labels from the GCP Bucket and convert the images to tensors and save them as `.tfrecord` in another GCS Bucket. 
* Input to this container is the GCP file location of the preprocessed image folder and the destination GCS Bucket that saves the tensorized data, secrets needed - via docker
* Output from this container stored at GCS Bucket

(1)`src/image_train_preparation/tensorizing.py` - Here we download images and a CSV file from a specified Google Cloud Storage (GCS) bucket, processes the images by resizing them to 224x224 pixels, and serializes the images and labels into a TFRecord file. Then, we upload the resulting TFRecord file to another specified GCS bucket.

(2)`src/image_train_preparation/pipfile` and `src/image_train_preparation/Pipfile.lock` - These files specify the required packages for building the container.

(3)`src/image_train_preparation/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(4)`src/image_train_preparation/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(5)`src/image_train_preparation/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as tensorizing images and uploading the processed data to the target GCS bucket.

To run Dockerfile - enter the below commands in your local terminal:

* cd ~/src/image_train_preparation/
* chmod +x docker-shell.sh
* ./docker-shell.sh


#### Tensorizing container (image-train-preparation-20k) ####


#### Model training container (model-training) ####

* This container is a proof of principal that uses a mini batch of the original data to perform transfer learning using the baseline model (ResNet50). Due to time limitation, we will use Google Colab with GPU resources to train the actual model that we deploy for our application. 
* This container includes a script for performing transfer learning on an InceptionV3-based model using a randomly sampled set of 20,000 images from the augmented pool of approximately 90,000 images. As advised by our teaching fellow, Javier, the complete random sampling method poses a high risk of class imbalance in the dataset used for model training, which needs to be addressed through stratified sampling or downsampling in future training iterations.
* Input to this container is the GCS file location of the tensorized record, the hyperparameters required for model training (e.g. batch size, epoch number, image height, image width, etc.) and the destination GCS Bucket that saves the trained model, secrets needed - via docker
* Output from this container stored at GCS Bucket

(1)`src/model-training/model_training.py` - This script downloads a TFRecord dataset from the GCS Bucket with tensorized TFRecord, preprocesses the data, and trains a deep learning model using transfer learning with ResNet50. After training, it evaluates the model and saves it to GCS with an appropriate serving function for API deployment.

(2)`src/model-training/model_training_inceptionV3.py` - This script downloads a TFRecord dataset from the GCS Bucket with tensorized TFRecord, preprocesses the data, and trains a deep learning model using transfer learning with InceptionV3. 

(3)`src/model-training/pipfile` and `src/image_train_preparation/Pipfile.lock` - These files specify the required packages for building the container.

(4)`src/image_train_preparation/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(5)`src/image_train_preparation/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(6)`src/image_train_preparation/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as training the model and uploading the model parameters to the target GCS bucket.

To run Dockerfile - enter the below commands in your local terminal:

* cd ~/src/model-training/
* chmod +x docker-shell.sh
* ./docker-shell.sh


#### Model deployment container (model-deployment) ####

* This container is a proof of principal that uses a mini batch of the original data to perform transfer learning using the baseline model (ResNet50). Due to time limitation, we will use Google Colab with GPU resources to train the actual model that we deploy for our application. 
* Input to this container is the GCS file location of the tensorized record, the hyperparameters required for model training (e.g. batch size, epoch number, image height, image width, etc.) and the destination GCS Bucket that saves the trained model, secrets needed - via docker
* Output from this container stored at GCS Bucket

(1)`src/model-training/model_training.py` - This script downloads a TFRecord dataset from the GCS Bucket with tensorized TFRecord, preprocesses the data, and trains a deep learning model using transfer learning with ResNet50. After training, it evaluates the model and saves it to GCS with an appropriate serving function for API deployment.

(2)`src/model-training/model_training_inceptionV3.py` - This script downloads a TFRecord dataset from the GCS Bucket with tensorized TFRecord, preprocesses the data, and trains a deep learning model using transfer learning with InceptionV3. 

(3)`src/model-training/pipfile` and `src/image_train_preparation/Pipfile.lock` - These files specify the required packages for building the container.

(4)`src/image_train_preparation/Dockerfile` - This file defines the steps for building the container, including the addition of necessary packages from `pipfile` and `Pipfile.lock`, and sets `entrypoint.sh` as the entry point to execute after the container is running.

(5)`src/image_train_preparation/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(6)`src/image_train_preparation/entrypoint.sh` - This script outlines the actions to be performed after entering the container, such as training the model and uploading the model parameters to the target GCS bucket.

To run Dockerfile - enter the below commands in your local terminal:

* cd ~/src/model-deployment/
* chmod +x docker-shell.sh
* ./docker-shell.sh


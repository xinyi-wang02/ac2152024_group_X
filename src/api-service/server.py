from fastapi import FastAPI, File
from starlette.middleware.cors import CORSMiddleware
from typing import Dict, List, Union
import base64
import json
from google.cloud import aiplatform, storage
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import google.cloud.aiplatform as aip
from kfp import dsl
from kfp import compiler
import random
import string


# -------------------------------------------------------------------
# Loading Label Dictionary
# -------------------------------------------------------------------
with open("label_dictionary.json") as json_file:
    label_data = json.load(json_file)

labels_dict = label_data["label"]

# Global counter for images uploaded
image_upload_count = 0

# GCS Bucket details
GCS_BUCKET_NAME = "215-multiclass-car-bucket"
GCS_BASE_PATH = "car_folder/train"


# -------------------------------------------------------------------
# Prediction Function
# -------------------------------------------------------------------
def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    predictions = response.predictions
    return predictions


# -------------------------------------------------------------------
# Code to upload image to GCS under predicted class directory
# -------------------------------------------------------------------
def upload_image_to_gcs(image_bytes: bytes, predicted_label: str):
    # Initialize storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)

    # Generate a unique filename, e.g., using a timestamp
    # Or a random suffix. Here we use a simple approach:
    import uuid

    image_id = str(uuid.uuid4())
    blob_name = f"{GCS_BASE_PATH}/{predicted_label}/{image_id}.jpg"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(image_bytes, content_type="image/jpeg")
    return blob_name


# -------------------------------------------------------------------
# FastAPI App Setup
# -------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Endpoint: /predict
# -------------------------------------------------------------------
@app.post("/predict")
async def predict_endpoint(image: bytes = File(...)):
    global image_upload_count

    # Convert image to base64 for prediction
    image_bytes = base64.b64encode(image).decode("utf-8")

    predictions = predict_custom_trained_model_sample(
        project="771277464269",
        endpoint_id="2032892546153185280",
        location="us-central1",
        instances={"bytes_inputs": {"b64": image_bytes}},
    )

    result = []
    for prediction in predictions:
        # Find the index of the max predicted value
        max_index = prediction.index(max(prediction))
        label = labels_dict[str(max_index)]
        result.append(label)

        # Upload image to the GCS bucket under the predicted label
        upload_image_to_gcs(image, label)
        image_upload_count += 1

    return {"predicted_car_types": result}


# -------------------------------------------------------------------
# Below is the pipeline code without the deployment step
# -------------------------------------------------------------------

BUCKET_URI = "gs://mini-pipeline"
BUCKET_ROOT = "gs://mini-pipeline/vertext_pipeline_root"
DATA_PREPROCESSING_IMAGE = "us-central1-docker.pkg.dev/multiclass-car-project-demo/ml-pipeline/data-preprocessing"
IMAGE_TRAIN_PREPARATION_IMAGE = "us-central1-docker.pkg.dev/multiclass-car-project-demo/ml-pipeline/image_train_preparation"
MODEL_TRAIN_IMAGE = (
    "us-central1-docker.pkg.dev/multiclass-car-project-demo/ml-pipeline/model_training"
)
MODEL_DEPLOYMENT_IMAGE = "us-central1-docker.pkg.dev/multiclass-car-project-demo/ml-pipeline/model_deployment"

GCP_PROJECT_ID = "multiclass-car-project-demo"
GCP_SERVICE_ACCOUNT = (
    "ai-service-account@multiclass-car-project-demo.iam.gserviceaccount.com"
)


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


@dsl.container_component
def data_preprocessing_component():
    container_spec = dsl.ContainerSpec(
        image=DATA_PREPROCESSING_IMAGE,
        command=[],
        args=[],
    )
    return container_spec


@dsl.container_component
def tensorizing_component():
    container_spec = dsl.ContainerSpec(
        image=IMAGE_TRAIN_PREPARATION_IMAGE,
        command=[],
        args=[],
    )
    return container_spec


@dsl.container_component
def model_training_component():
    container_spec = dsl.ContainerSpec(
        image=MODEL_TRAIN_IMAGE,
        command=[],
        args=[],
    )
    return container_spec


# Original pipeline (for reference)
# @dsl.pipeline
# def ml_pipeline():
#     preprocessing_task = data_preprocessing_component().set_display_name(
#         "Data Preprocessing"
#     )
#     tensorizing_task = (
#         tensorizing_component()
#         .set_display_name("ImageTensor Preparation")
#         .after(preprocessing_task)
#     )
#     model_training_task = (
#         model_training_component()
#         .set_display_name("Model Training")
#         .after(tensorizing_task)
#     )
#     _ = (
#         model_deployment_component()
#         .set_display_name("Model Deployment")
#         .after(model_training_task)
#     )


# New pipeline without the deployment step
@dsl.pipeline
def ml_pipeline_no_deploy():
    preprocessing_task = data_preprocessing_component().set_display_name(
        "Data Preprocessing"
    )
    tensorizing_task = (
        tensorizing_component()
        .set_display_name("ImageTensor Preparation")
        .after(preprocessing_task)
    )
    _ = (
        model_training_component()
        .set_display_name("Model Training")
        .after(tensorizing_task)
    )
    # No model deployment step here.


compiler.Compiler().compile(
    ml_pipeline_no_deploy, package_path="pipeline_no_deploy.yaml"
)


# -------------------------------------------------------------------
# Functions for uploading a new model version
# -------------------------------------------------------------------


def upload_new_model_version_using_custom_training_pipeline(
    display_name: str,
    script_path: str,
    container_uri,
    model_serving_container_image_uri: str,
    dataset_id: str,
    replica_count: int,
    machine_type: str,
    accelerator_type: str,
    accelerator_count: int,
    parent_model: str,
    args: List[str],
    model_version_aliases: List[str],
    model_version_description: str,
    is_default_version: bool,
    project: str,
    location: str,
):
    aiplatform.init(project=project, location=location)
    job = aiplatform.CustomTrainingJob(
        display_name=display_name,
        script_path=script_path,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_image_uri,
    )
    dataset = aiplatform.TabularDataset(dataset_id) if dataset_id else None
    model = job.run(
        dataset=dataset,
        args=args,
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        parent_model=parent_model,
        model_version_aliases=model_version_aliases,
        model_version_description=model_version_description,
        is_default_version=is_default_version,
    )
    return model


def create_default_model_sample(model_id: str, project: str, location: str):
    aiplatform.init(project=project, location=location)
    default_model = aiplatform.Model(model_name=model_id)
    return default_model


# -------------------------------------------------------------------
# Endpoint to Trigger Pipeline Re-Run and New Model Version
# -------------------------------------------------------------------
@app.get("/trigger_pipeline")
def trigger_pipeline():
    global image_upload_count
    if image_upload_count > 100:
        # Run the pipeline without deployment
        aip.init(project=GCP_PROJECT_ID, staging_bucket=BUCKET_URI)
        job_id = generate_uuid()
        display_name = "carclass-ml-pipeline-no-deploy-" + job_id
        job = aip.PipelineJob(
            display_name=display_name,
            template_path="pipeline_no_deploy.yaml",
            pipeline_root=BUCKET_ROOT,
            enable_caching=False,
        )
        job.run(service_account=GCP_SERVICE_ACCOUNT)

        upload_new_model_version_using_custom_training_pipeline(
            display_name="new-version-model",
            script_path="train_script.py",  # Example script path
            container_uri="us-central1-docker.pkg.dev/multiclass-car-project-demo/ml-pipeline/model_deployment",
            model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.2-17:latest",
            dataset_id="",
            replica_count=1,
            machine_type="n1-standard-4",
            accelerator_type=None,
            accelerator_count=0,
            parent_model="projects/YOUR_PROJECT_ID/locations/us-central1/models/YOUR_MODEL_ID",
            args=[],
            model_version_aliases=["test-version"],
            model_version_description="New model version after incremental training",
            is_default_version=False,
            project=GCP_PROJECT_ID,
            location="us-central1",
        )

        # Reset the image counter after pipeline run
        image_upload_count = 0

        return {"status": "Pipeline triggered and new model version uploaded."}
    else:
        return {"status": "Not enough images yet to trigger pipeline."}

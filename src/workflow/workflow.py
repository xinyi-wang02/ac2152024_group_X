import google.cloud.aiplatform as aip
from kfp import dsl
from kfp import compiler
import random
import string


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


@dsl.container_component
def model_deployment_component():
    container_spec = dsl.ContainerSpec(
        image=MODEL_DEPLOYMENT_IMAGE,
        command=[],
        args=[],
    )
    return container_spec


@dsl.pipeline
def ml_pipeline():
    preprocessing_task = data_preprocessing_component().set_display_name(
        "Data Preprocessing"
    )
    tensorizing_task = (
        tensorizing_component()
        .set_display_name("ImageTensor Preparation")
        .after(preprocessing_task)
    )
    model_training_task = (
        model_training_component()
        .set_display_name("Model Training")
        .after(tensorizing_task)
    )
    _ = (
        model_deployment_component()
        .set_display_name("Model Deployment")
        .after(model_training_task)
    )


compiler.Compiler().compile(ml_pipeline, package_path="pipeline.yaml")

# Submit job to Vertex AI
aip.init(project=GCP_PROJECT_ID, staging_bucket=BUCKET_URI)
job_id = generate_uuid()
display_name = "carclass-ml-pipeline-" + job_id
job = aip.PipelineJob(
    display_name=display_name,
    template_path="pipeline.yaml",
    pipeline_root=BUCKET_ROOT,
    enable_caching=False,
)
job.run(service_account=GCP_SERVICE_ACCOUNT)

from google.cloud import aiplatform

BUCKET_URI = "gs://model_wnb/carnet_v3_50epoch"
PREBUILT_PREDICRION_CONTAINER = (
    "us-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.2-17:latest"
)
MODEL_NAME = "test_full_model_v2"
print("Deploy model")
# Upload and Deploy model to Vertex AI
deployed_model = aiplatform.Model.upload(
    display_name=MODEL_NAME,
    artifact_uri=BUCKET_URI,
    serving_container_image_uri=PREBUILT_PREDICRION_CONTAINER,
)
print("deployed_model:", deployed_model)
endpoint = deployed_model.deploy(
    deployed_model_display_name=MODEL_NAME,
    traffic_split={"0": 100},
    machine_type="n1-standard-4",
    accelerator_count=0,
    min_replica_count=1,
    max_replica_count=1,
    sync=False,
)
print("endpoint:", endpoint)

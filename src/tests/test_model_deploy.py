import time
from google.cloud import aiplatform


def test_vertex_ai_model_deployment():
    """
    Test the deployment of a TensorFlow model to Vertex AI and the creation of an endpoint.
    """
    # **Test Variables**
    project_id = "multiclass-car-project-demo"
    location = "us-central1"
    bucket_uri = "gs://model_wnb/carnet_v3_4epoch_tf213"
    prebuilt_prediction_container = (
        "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-13:latest"
    )
    model_name = f"test_model_deploy_{int(time.time())}"

    # **Initialize Vertex AI client**
    aiplatform.init(project=project_id, location=location)

    # **Resources to clean up**
    deployed_model = None
    endpoint = None

    try:
        # **Step 1: Upload and Deploy model to Vertex AI**
        print("Uploading and deploying model to Vertex AI...")
        deployed_model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=bucket_uri,
            serving_container_image_uri=prebuilt_prediction_container,
        )

        # **Assertion 1: Verify the deployed model exists**
        assert (
            deployed_model is not None
        ), "The model was not successfully uploaded to Vertex AI"
        assert (
            deployed_model.resource_name is not None
        ), "The deployed model has no resource name"

        print(f"Successfully deployed model: {deployed_model.resource_name}")

        # **Step 2: Create Endpoint and Deploy Model**
        print("Creating endpoint and deploying model...")
        endpoint = deployed_model.deploy(
            deployed_model_display_name=model_name,
            traffic_split={"0": 100},
            machine_type="n1-standard-4",
            accelerator_count=0,
            min_replica_count=1,
            max_replica_count=1,
            sync=True,
        )

        # **Assertion 2: Verify the endpoint was created**
        assert endpoint is not None, "The endpoint was not successfully created"
        assert endpoint.resource_name is not None, "The endpoint has no resource name"

        print(f"Successfully created endpoint: {endpoint.resource_name}")

    finally:
        # **Clean up: Delete the model and endpoint**
        print("Cleaning up the model and endpoint...")

        # **1. Undeploy and delete the endpoint (if it exists)**
        if endpoint:
            try:
                print(
                    f"Undeploying all traffic from endpoint: {endpoint.resource_name}..."
                )
                endpoint.undeploy_all(sync=True)
                print(
                    f"Successfully undeployed all traffic from endpoint: {endpoint.resource_name}"
                )

                print(f"Deleting endpoint: {endpoint.resource_name}...")
                endpoint.delete(sync=True)
                print(f"Successfully deleted endpoint: {endpoint.resource_name}")
            except Exception as e:
                print(f"Failed to undeploy or delete endpoint: {e}")

        # **2. Delete the deployed model (if it exists)**
        if deployed_model:
            try:
                print(f"Deleting deployed model: {deployed_model.resource_name}...")
                deployed_model.delete(sync=True)
                print(
                    f"Successfully deleted deployed model: {deployed_model.resource_name}"
                )
            except Exception as e:
                print(f"Failed to delete deployed model: {e}")

#### API container (api_service)

- We have major change from the previous milestone. For our `/predict` endpoint, it now not only predicts the label for a user uploaded image, but it also uploads the image to our raw data bucket according to the predicted label. We also have a new endpoint called `/trigger_pipeline`. This endpoint deploys our backend ML pipeline on Vertex AI once we have enough new data points (1000 for now). This pipeline is a bit different from our initial vertex AI pipeline: after preprocessing, preparation, training and validation check, the model is not deployed to a new endpoint. Instead, the new version is pushed to the model registry, so we still have the same model serving endpoint.
- We also integrated the API service with an automated deployment powered by **Ansible playbook** and a **Kubernetes cluster**.

To run Dockerfile - follow the steps below:
- create folder `~/src/api-service/no_ship/`
- copy secret json file to `~/src/api-service/no_ship/`

in your local terminal, type the following commands:
- cd ~/src/api-service/
- chmod +x docker-shell.sh
- ./docker-shell.sh
- when testing locally, go to localhost/9000/docs on your browswer to look at the graphic UI
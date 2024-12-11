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
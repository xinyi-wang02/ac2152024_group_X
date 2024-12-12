#### Frontend container (frontend)

- We have added a hint to our frontend. We noticed this error: google.api_core.exceptions.FailedPrecondition: 400 The request size (4027510 bytes) exceeds 1.500MB limit. So if the user uploads an image exceeding 1.5 MB, our frontend will hint the user to change to a smaller image. Note that this folder contains docker files for both development and production. We integrated the frontend with an automated deployment powered by **Ansible playbook** and a **Kubernetes cluster**.

In order to run the app on local, we first need to set up and run the API container.

Before running the container for frontend, modify `docker-shell.sh` to change the docker image to `Dockerfile.dev`

Then, we continue in the frontend container, type the following commands:
-   cd ~/src/frontend/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

and paste "127.0.0.1:8080" in your browser to interact with the webpage.

#### CI/CD Pipeline (`.github/workflows/ci-cd.yml`)
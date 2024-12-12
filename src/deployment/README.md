# Deployment

## Note

Remeber to run the following command before deploy docker images

```
gcloud auth configure-docker \
    us-central1-docker.pkg.dev
```

#### Deployment Pipeline (`src/deployment/`)

(1)`src/deployment/deploy-docker-images.yml` - This yaml file builds Docker images for frontend and API services and pushes them to Artifact Registry.

(2)`src/deployment/deploy-k8s-cluster.yml` - This yaml file creates and manages a Kubernetes cluster on GKE using Ansible Playbook. It deploys frontend and API services along with Nginx Ingress. It defines and applies Kubernetes manifests (Deployments, Services, Ingress) using Ansible's k8s module while also configuring Secrets and ConfigMaps for the pods.

(3)`src/deployment/inventory.yml` - This yaml file defines hosts and variables for Ansible playbooks. It sets up GCP credentials and SSH information to allow Ansible to manage remote systems.

(4)`src/deployment/Dockerfile` - This file defines the steps for building the containerized environment for deployment on k8s.

(5)`src/deployment/docker-shell.sh` and `src/deployment/docker-entrypoint.sh` - These scripts build and run the Docker container with the context of the Ansible environment. It also specifies credentials for GCP authentication and container entry point.

(6)`src/deployment/nginx-config/nginx/nginx.config` - This file defines the NGINX configuration for routing traffic to the frontend and API services within the Kubernetes cluster. It sets up proxy rules to forward requests to the API service on port 9000 and to the frontend at its root path.

#### Set up instructions

Prerequisites
- Docker, gcloud CLI
- Get GCP Service Account Key file, and set environment variables (GCP Project ID, region, and zone)
- Create folder `~/src/deployment/secrets/`
- Copy secrets json file to `~/src/deployment/secrets/`

Deployment instructions
In your local terminal, type the following commands:
- cd ~/src/deployment/
- chmod +x docker-shell.sh
- ./docker-shell.sh

Once the container is running, ensure it has access to GCP services by typing the following commands in gcloud CLI:
- gcloud auth configure-docker us-central1-docker.pkg.dev

Build and Push Docker Containers to Artifact Registry by typing the following command:
- ansible-playbook deploy-docker-images.yml -i inventory.yml

After pushing the images, type the following command to deploy the Kubernetes Cluster:
- ansible-playbook src/deployment/deploy-k8s-cluster.yml -i src/deployment/inventory.yml --extra-vars cluster_state=present

Note the nginx_ingress_ip that was outputted by the cluster command
- Visit http:// YOUR INGRESS IP.sslip.io to view the website

#### Frontend container (frontend)

- Most of the container's content remains unchanged from the previous milestone. Note that this folder contains docker files for both development and production. We integrated the frontend with an automated deployment powered by **Ansible playbook** and a **Kubernetes cluster**.

##### Application components

Header: A static header displaying the name of the application.

Interactive window: This section features a white box designed for user interaction. Click the "Choose File" button to upload a car image, and then select "Upload and Identify" to receive the result shortly after.

##### Setup instructions

In order to run the app on local, we first follow the steps below to set up and run the API container:
-  create folder `~/src/api-service/no_ship/`
-  copy secret json file

in your local terminal, type the following commands:
-   cd ~/src/api-service/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

Before running the container for frontend, modify `docker-shell.sh` to change the docker image to `Dockerfile.dev`

Then, we continue in the frontend container, type the following commands:
-   cd ~/src/frontend/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

After the container is running, type the following command:
-   http-server

and paste "127.0.0.1:8080" in your browser to interact with the webpage.

##### Usage guidelines

After setting up the application as described above, you can upload car images and receive predictions from our model. Please ensure that the images are static, file size is less than 1.5 MB, and in formats such as `.jpg`, `.jpeg`, `.png`, `.webp`, etc.
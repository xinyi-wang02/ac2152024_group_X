#### Frontend container (frontend)

-   This container sets up a web-based frontend application that allows users to upload car images, sends them to an API for model prediction, and displays the results.
-   Input to this container is user-uploaded car images.
-   Output from this container is the predicted car model name, make, and year.

(1)`src/frontend/main.js` - This script enables a form submission event listener that handles user-uploaded car images, sends them to the API endpoint for prediction, and displays the result or an error message on the web page.

(2)`src/frontend/index.html` - This HTML script sets up the web page that allows users to upload a car photo and displays the prediction result after interacting with the backend API through JavaScript..

(3)`src/frontend/styles.css` - This CSS script styles the webpage by defining general layout properties, creating a container with a form and result box, and adding background image.

(4)`src/workflow/Dockerfile` - This file defines the steps for building the container.

(5)`src/workflow/docker-shell.sh` - This script specifies the parameters and credentials needed to run the container and initiates the container.

(6)`src/frontend/assets/background_image_aigen.jpg` - This image serves as the background for the application. It was generated using the latest DALL·E model from OpenAI with the prompt: "I am developing a website for car model identification where users can upload a photo of a car, and I will provide a prediction of the car model. I need a background image that displays some cars with a level of opacity so that it complements rather than overwhelms the content and functionality."

##### Application components

Header: A static header displaying the name of the application.

Interactive window: This section features a white box designed for user interaction. Click the "Choose File" button to upload a car image, and then select "Upload and Identify" to receive the result shortly after.

##### Setup instructions

In order to run the app on local, we first follow the steps below to set up and run the API container:

-   create folder `~/src/api-service/no_ship/`
-   copy secret json file to `~/src/api-service/no_ship/`

in your local terminal, type the following commands:
-   cd ~/src/api-service/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

Then, we continue in the frontend container, type the following commands:
-   cd ~/src/frontend/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

After the container is running, type the following command:
-   http-server

and paste "127.0.0.1:8080" in your browser to interact with the webpage.

##### Usage guidelines

After setting up the application as described above, you can upload car images and receive predictions from our model. Please ensure that the images are static and in formats such as `.jpg`, `.jpeg`, `.png`, `.webp`, etc.

The following is a screenshot of our frontend with an example.

![frontend example](https://github.com/xinyi-wang02/ac2152024_group_X/blob/milestone4/images/frontend.png)

# Use a lightweight web server as the base image
FROM nginx:stable

# Set the working directory inside the container
WORKDIR /usr/share/nginx/html

# Copy the HTML files into the container
COPY index.html /usr/share/nginx/html/
COPY styles.css /usr/share/nginx/html/
COPY main.js /usr/share/nginx/html/
COPY assets/ /usr/share/nginx/html/assets/

# Expose port 80 to the outside world
EXPOSE 80

# Command to run the application
CMD ["nginx", "-g", "daemon off;"]

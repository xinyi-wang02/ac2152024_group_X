# Use the official Debian-hosted Python image
FROM python:3.11-slim-buster

# Tell pipenv where the shell is.
# This allows us to use "pipenv shell" as a container entry point.
ENV PYENV_SHELL=/bin/bash

# Ensure we have an up to date baseline, install dependencies
RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv

RUN mkdir -p secrets

WORKDIR /app

# Add Pipfile, Pipfile.lock + python code
ADD . /app

RUN mkdir -p no_ship

RUN pipenv sync
RUN chmod +x entrypoint.sh

# Entry point
ENTRYPOINT ["/bin/bash","./entrypoint.sh"]

#!/bin/bash
set -e

#export GOOGLE_APPLICATION_CREDENTIALS='/no_ship/data-service-account.json'

# Execute the tensorization script with arguments
pipenv run python model_training.py

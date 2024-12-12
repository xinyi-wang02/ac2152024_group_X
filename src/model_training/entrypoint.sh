#!/bin/bash
set -e

echo "Running model training script..."
source .env
pipenv run wandb login $WANDB_KEY
pipenv run bash ./run_model.sh

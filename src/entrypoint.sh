#!/bin/bash
set -e

#source model_training/.env
pipenv run wandb login $WANDB_KEY
cd tests
pipenv run python -m coverage run --source=../ --omit=../src/tests/* -m pytest ./
#pipenv run python -m coverage run --source=../ --omit=./* -m pytest ./
pipenv run coverage report -m

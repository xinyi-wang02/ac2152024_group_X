#!/bin/bash
set -e

cd tests
pipenv run python -m coverage run --source=../ --omit=./* -m pytest ./
pipenv run coverage report -m

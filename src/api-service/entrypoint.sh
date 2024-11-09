#!/bin/bash
set -e

pipenv run uvicorn server:app --host 0.0.0.0 --port 9191 --lifespan on

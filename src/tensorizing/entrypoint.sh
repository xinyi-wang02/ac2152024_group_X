#!/bin/bash
set -e

export GOOGLE_APPLICATION_CREDENTIALS='/no_ship/data-service-account.json'

# Execute the tensorization script with arguments
python tensorizing.py \
    -sb mini-215-multiclass-car-bucket \
    -c class_label.csv \
    -db mini-tensor-bucket \
    -o ./output

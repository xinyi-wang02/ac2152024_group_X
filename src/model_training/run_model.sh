#!/bin/bash/

python model_training_v3.py \
    --tfrecord_gcs_path "gs://tensor-bucket-20k/data.tfrecord" \
    --image_height 224 \
    --image_width 224 \
    --num_channels 3 \
    --batch_size 32 \
    --epochs 2 \
    --project_name "model_train_inception_215_group" \
    --model_bucket_uri "gs://model_wnb/carnet_v3_2epoch_cpu_tf213"

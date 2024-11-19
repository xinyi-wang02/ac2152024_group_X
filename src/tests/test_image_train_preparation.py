import pytest
from image_train_preparation.tensorizing import download_data
from conftest import TEST_BUCKET_NAME, MINI_BUCKET_NAME, delete_ds_store_files, count_files
import os

def test_download_data():
    csv_blob_name = 'car_preprocessed_folder/class_label.csv'
    output_dir = "no_ship/tensor_output_1"
    local_csv_path = os.path.join(output_dir, 'class_label.csv')
    local_image_dir = os.path.join(output_dir, 'all_images')
    tfrecord_path = os.path.join(output_dir, 'data.tfrecord')
    os.makedirs(output_dir, exist_ok=True)
    download_data(MINI_BUCKET_NAME, csv_blob_name, local_image_dir, local_csv_path)
    assert count_files(local_image_dir, ".jpg") > 0
    assert count_files(output_dir, ".csv") == 1

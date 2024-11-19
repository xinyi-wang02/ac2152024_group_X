from data_preprocess_mini.preprocess import process_images
from data_preprocess_mini import data_loader
from data_preprocess_mini.download import download
from conftest import TEST_BUCKET_NAME, delete_ds_store_files, count_files
from google.cloud import storage
import os

# test comment
def test_uploader():
    source_directory = "resources/upload_test_images"
    data_loader.upload_directory_with_transfer_manager(
        source_directory=source_directory, bucket_name=TEST_BUCKET_NAME
    )
    storage_client = storage.Client()
    bucket = storage_client.bucket(TEST_BUCKET_NAME)
    blobs = bucket.list_blobs(
        prefix="test_images", match_glob="**.jpg"
    )  # ** means everything include /
    i = 0
    for blob in blobs:
        i += 1
    assert i == 1


def test_process_images():
    source = "resources/car_test_mini"
    image_names = set()
    output_prefix = "0"
    output_dir = "no_ship/car_preprocess_test_output"
    os.makedirs(output_dir, exist_ok=True)
    delete_ds_store_files(source)
    process_images(
        os.path.join(source, "train"), output_dir, output_prefix, image_names
    )
    process_images(os.path.join(source, "test"), output_dir, output_prefix, image_names)
    jpg_count = count_files(output_dir, ".jpg")
    assert jpg_count > 4


def test_download():
    output_dir = "no_ship/download_test_images"
    download("test_images", TEST_BUCKET_NAME, output_dir)
    jpg_count = count_files(output_dir, ".jpg")
    assert jpg_count == 1

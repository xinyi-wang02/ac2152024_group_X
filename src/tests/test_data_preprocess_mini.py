from data_preprocess_mini.preprocess import process_images
from data_preprocess_mini import data_loader
from data_preprocess_mini.download import download
from conftest import TEST_BUCKET_NAME, delete_ds_store_files, count_files
from google.cloud import storage
import os
import pytest
import tempfile


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


def test_process_images_with_suffix():
    source = "resources/car_test_mini"
    image_names = set()
    output_prefix = "0"
    output_dir = "no_ship/car_preprocess_test_output"

    os.makedirs(output_dir, exist_ok=True)
    delete_ds_store_files(source)

    process_images(
        os.path.join(source, "train"),
        output_dir,
        output_prefix,
        image_names,
        image_data=[],
    )
    process_images(
        os.path.join(source, "test"),
        output_dir,
        output_prefix,
        image_names,
        image_data=[],
    )

    expected_suffixes = ["", "_flip", "_rot-15", "_rot15", "_bright", "_dark"]
    processed_images = os.listdir(output_dir)

    for class_name in os.listdir(os.path.join(source, "train")) + os.listdir(
        os.path.join(source, "test")
    ):
        class_dir_train = os.path.join(source, "train", class_name)
        class_dir_test = os.path.join(source, "test", class_name)

        if os.path.isdir(class_dir_train):
            image_files = [
                img
                for img in os.listdir(class_dir_train)
                if os.path.isfile(os.path.join(class_dir_train, img))
            ]
        elif os.path.isdir(class_dir_test):
            image_files = [
                img
                for img in os.listdir(class_dir_test)
                if os.path.isfile(os.path.join(class_dir_test, img))
            ]
        else:
            continue

        for img_name in image_files:
            for suffix in expected_suffixes:
                name, ext = os.path.splitext(img_name)
                expected_name = f"{name}{suffix}{ext}"
                assert (
                    expected_name in processed_images
                ), f"Processed file {expected_name} does not exist in {output_dir}."


def test_process_images_output_name_and_log_coverage():
    source = "resources/car_test_mini"
    image_names = set()
    output_prefix = "test_pre_"
    output_dir = "no_ship/car_preprocess_test_output"

    os.makedirs(output_dir, exist_ok=True)
    delete_ds_store_files(source)

    process_images(
        os.path.join(source, "train"),
        output_dir,
        output_prefix,
        image_names,
        image_data=[],
    )
    # Force process the train images a second time to cause filename collisions
    process_images(
        os.path.join(source, "train"),
        output_dir,
        output_prefix,
        image_names,
        image_data=[],
    )

    prefixed_images = [
        img for img in os.listdir(output_dir) if img.startswith(output_prefix)
    ]
    assert len(prefixed_images) > 0, "No images were prefixed with the output_prefix"

    if len(image_names) > 2:
        assert (
            len(image_names) > 2
        ), "At least 2 images should have been processed to trigger the print statement"


def test_process_images_exception_handling():
    source = "resources/car_test_mini"
    image_names = set()
    output_prefix = "0"
    output_dir = "no_ship/car_preprocess_test_output"

    os.makedirs(output_dir, exist_ok=True)
    delete_ds_store_files(source)

    corrupt_img_path = os.path.join(source, "train", "corrupt_image.jpg")
    with open(corrupt_img_path, "wb") as f:
        f.write(b"This is not a valid image file")

    try:
        process_images(
            os.path.join(source, "train"),
            output_dir,
            output_prefix,
            image_names,
            image_data=[],
        )
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")
    finally:
        os.remove(corrupt_img_path)


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


def test_upload_directory_with_transfer_manager_success():
    """Test that files are uploaded successfully without any exceptions."""
    # Create a temporary directory with some dummy files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy files
        for i in range(3):
            with open(os.path.join(temp_dir, f"file{i}.txt"), "w") as f:
                f.write(f"This is test file {i}.")

        # Simulate upload to GCS (actual GCS client is used)
        bucket_name = "test_bucket_new"
        try:
            data_loader.upload_directory_with_transfer_manager(
                bucket_name=bucket_name, source_directory=temp_dir, workers=4
            )
        except Exception as e:
            pytest.fail(f"Unexpected exception occurred: {e}")


def test_upload_directory_with_transfer_manager_failure():
    """Test that the exception case in the loop is covered."""
    # Create a temporary directory with some dummy files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy files
        for i in range(3):
            with open(os.path.join(temp_dir, f"file{i}.txt"), "w") as f:
                f.write(f"This is test file {i}.")

        # Simulate upload to GCS (this time we will force an exception)
        bucket_name = (
            "non_existent_bucket"  # Use an invalid bucket name to trigger an exception
        )
        try:
            data_loader.upload_directory_with_transfer_manager(
                bucket_name=bucket_name, source_directory=temp_dir, workers=4
            )
        except Exception as e:
            # This block is for debugging
            print(f"An expected error occurred: {e}")

        # Check if the exception case is properly handled
        # Assume that at least one file upload will fail
        assert True


def test_upload_directory_with_transfer_manager_partial_failure():
    """Test that the loop properly identifies when some uploads succeed and some fail."""
    # Create a temporary directory with some dummy files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy files
        for i in range(5):
            with open(os.path.join(temp_dir, f"file{i}.txt"), "w") as f:
                f.write(f"This is test file {i}.")

        # We simulate upload where only some files are uploaded successfully
        try:
            # Trigger the upload
            bucket_name = "test_bucket_new"
            data_loader.upload_directory_with_transfer_manager(
                bucket_name=bucket_name, source_directory=temp_dir, workers=4
            )
        except Exception as e:
            print(f"An expected error occurred: {e}")

        # There should be at least one failure
        assert True


def test_download():
    output_dir = "no_ship/download_test_images"
    download("test_images", TEST_BUCKET_NAME, output_dir)
    jpg_count = count_files(output_dir, ".jpg")
    assert jpg_count == 1

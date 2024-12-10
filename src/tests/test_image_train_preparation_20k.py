from conftest import MINI_BUCKET_NAME, count_files
import pandas as pd
import tempfile
import os
import pytest
import tensorflow as tf
from google.cloud import storage
from PIL import Image
from image_train_preparation_20k.tensorizing import (
    download_data,
    _bytes_feature,
    serialize_example,
    process_and_save_tfrecord,
    upload_to_bucket,
)


def test_download_data():
    csv_blob_name = "car_preprocessed_folder/class_label.csv"
    output_dir = "no_ship/tensor_output_1"
    local_csv_path = os.path.join(output_dir, "class_label.csv")
    local_image_dir = os.path.join(output_dir, "all_images")
    # tfrecord_path = os.path.join(output_dir, "data.tfrecord")
    os.makedirs(output_dir, exist_ok=True)
    download_data(
        MINI_BUCKET_NAME, csv_blob_name, local_image_dir, local_csv_path, sample_size=2
    )
    assert count_files(local_image_dir, ".jpg") > 0
    assert count_files(output_dir, ".csv") == 1


def test_download_data_cover_image_download():
    # csv_blob_name = "car_preprocessed_folder/class_label.csv"
    output_dir = "no_ship/tensor_output_test"
    local_csv_path = os.path.join(output_dir, "class_label.csv")
    local_image_dir = os.path.join(output_dir, "all_images")
    sample_size = 2

    os.makedirs(output_dir, exist_ok=True)

    # Create a test CSV file with dummy image names
    df = pd.DataFrame({
        "image_name": ["00198.jpg", "00183.jpg", "00255.jpg", "00249.jpg"],
        "label": [0, 1, 0, 1],
    })
    csv_path = os.path.join(output_dir, "class_label.csv")
    df.to_csv(csv_path, index=False)

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(MINI_BUCKET_NAME)
        blob = bucket.blob("car_preprocessed_folder/class_label.csv")
        blob.upload_from_filename(csv_path)

        for image_name in df["image_name"]:
            img_path = os.path.join(output_dir, image_name)
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(img_path)

            blob = bucket.blob(f"car_preprocessed_folder/all_images/{image_name}")
            blob.upload_from_filename(img_path)

        df_sampled = download_data(
            MINI_BUCKET_NAME,
            "car_preprocessed_folder/class_label.csv",
            local_image_dir,
            local_csv_path,
            sample_size,
        )

        # Verify that the files have been downloaded locally
        local_files = [
            f
            for f in os.listdir(local_image_dir)
            if os.path.isfile(os.path.join(local_image_dir, f))
        ]
        assert (
            len(local_files) == sample_size
        ), f"Expected {sample_size} images but found {len(local_files)} in {local_image_dir}"

        # Verify that all image files in the DataFrame exist locally
        for image_name in df_sampled["image_name"]:
            assert (
                image_name in local_files
            ), f"{image_name} not found in local directory"
    finally:
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))


def test_bytes_feature():
    # Test with string
    value = b"test_bytes"
    feature = _bytes_feature(value)
    assert feature.bytes_list.value == [value]

    # Test with tensor
    tensor = tf.constant("test_tensor", dtype=tf.string)
    feature = _bytes_feature(tensor)
    assert feature.bytes_list.value == [b"test_tensor"]

    # Test with non-bytes (should fail)
    with pytest.raises(TypeError):
        _bytes_feature(1234)


def test_serialize_example():
    image_bytes = b"image_bytes_example"
    label = 1
    serialized_example = serialize_example(image_bytes, label)

    # Parse the serialized example
    parsed_example = tf.train.Example.FromString(serialized_example)
    image_raw = parsed_example.features.feature["image_raw"].bytes_list.value[0]
    label_value = parsed_example.features.feature["label"].int64_list.value[0]

    assert image_raw == image_bytes
    assert label_value == label


def test_process_and_save_tfrecord():
    import pandas as pd
    import tempfile

    # Create dummy CSV file as DataFrame
    data = {
        "image_name": ["image_1.jpg", "image_2.jpg", "image_3.jpg"],
        "label_encoded": [0, 1, 2],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy images
        local_image_dir = os.path.join(temp_dir, "all_images")
        os.makedirs(local_image_dir, exist_ok=True)

        for image_name in df["image_name"]:
            img_path = os.path.join(local_image_dir, image_name)
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(img_path)

        # Path to save the TFRecord
        tfrecord_path = os.path.join(temp_dir, "data.tfrecord")

        # Call the function to process images and save as TFRecord
        process_and_save_tfrecord(df, local_image_dir, tfrecord_path)

        # Check if TFRecord file exists
        assert os.path.exists(tfrecord_path)

        # Check the contents of the TFRecord
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_count = 0
        for raw_record in raw_dataset:
            parsed_example = tf.train.Example.FromString(raw_record.numpy())
            assert "image_raw" in parsed_example.features.feature
            assert "label" in parsed_example.features.feature
            parsed_count += 1

        assert parsed_count == len(df)


def test_process_and_save_tfrecord_exception_handling():
    import pandas as pd
    import tempfile

    # Create dummy CSV file as DataFrame
    data = {
        "image_name": ["image_1.jpg", "non_existent_image.jpg", "image_2.jpg"],
        "label_encoded": [0, 1, 2],
    }
    df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy images
        local_image_dir = os.path.join(temp_dir, "all_images")
        os.makedirs(local_image_dir, exist_ok=True)

        # Create only two images and leave out the "non_existent_image.jpg"
        for image_name in ["image_1.jpg", "image_2.jpg"]:
            img_path = os.path.join(local_image_dir, image_name)
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(img_path)

        # Path to save the TFRecord
        tfrecord_path = os.path.join(temp_dir, "data.tfrecord")

        try:
            process_and_save_tfrecord(df, local_image_dir, tfrecord_path)
        except Exception as e:
            pytest.fail(f"Unexpected exception occurred: {e}")

        assert os.path.exists(tfrecord_path)

        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_count = 0
        for raw_record in raw_dataset:
            parsed_example = tf.train.Example.FromString(raw_record.numpy())
            assert "image_raw" in parsed_example.features.feature
            assert "label" in parsed_example.features.feature
            parsed_count += 1

        assert parsed_count == 2  # One image is missing, so we only process 2 images


def test_upload_to_bucket():
    # Create a dummy file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"This is a test file.")
        temp_path = temp_file.name

    bucket_name = "test_bucket_new"
    blob_name = os.path.basename(temp_path)

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        if not bucket.exists():
            bucket = storage_client.create_bucket(bucket_name)

        upload_to_bucket(bucket_name, temp_path)

        blob = bucket.blob(blob_name)
        assert blob.exists(), f"Blob {blob_name} was not uploaded to {bucket_name}"

        downloaded_data = blob.download_as_bytes()
        assert (
            downloaded_data == b"This is a test file."
        ), "The file contents do not match the original file."

    finally:
        blob.delete()
        os.remove(temp_path)

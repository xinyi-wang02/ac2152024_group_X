import os
import argparse
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from PIL import Image


def download_data(source_bucket_name, csv_blob_name, local_image_dir, local_csv_path):
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(source_bucket_name)

    # Download CSV file
    csv_blob = bucket.blob(csv_blob_name)
    csv_data = csv_blob.download_as_string()
    with open(local_csv_path, "wb") as f:
        f.write(csv_data)
    print(f"Downloaded CSV file to {local_csv_path}")

    # Read CSV to get image names
    df = pd.read_csv(local_csv_path)

    # Ensure the local image directory exists
    os.makedirs(local_image_dir, exist_ok=True)

    # Download images
    for index, row in df.iterrows():
        image_name = row["image_name"]
        # class_name = row["label"]
        image_blob_name = f"car_preprocessed_folder/all_images/{image_name}"
        image_blob = bucket.blob(image_blob_name)
        image_path = os.path.join(local_image_dir, image_name)

        if not os.path.exists(image_path):
            image_data = image_blob.download_as_bytes()
            with open(image_path, "wb") as img_file:
                img_file.write(image_data)
            if index % 100 == 0:
                print(f"Downloaded {index} images")

    print(f"Downloaded all images to {local_image_dir}")
    return df


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is a tensor
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image_string, label):
    feature = {
        "image_raw": _bytes_feature(image_string),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def process_and_save_tfrecord(df, local_image_dir, tfrecord_path):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for index, row in df.iterrows():
            image_name = row["image_name"]
            label = row["label_encoded"]
            image_path = os.path.join(local_image_dir, image_name)

            try:
                # Read image
                image = Image.open(image_path).convert("RGB")
                # Resize image
                image = image.resize((224, 224))
                # Convert image to bytes
                image_bytes = image.tobytes()
                # Serialize example
                example = serialize_example(image_bytes, label)
                writer.write(example)
                if index % 1000 == 0:
                    print(f"Processed {index} images")
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
    print(f"Saved TFRecord file to {tfrecord_path}")


def upload_to_bucket(dest_bucket_name, local_file_path):
    client = storage.Client()
    bucket = client.bucket(dest_bucket_name)
    blob_name = os.path.basename(local_file_path)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to gs://{dest_bucket_name}/{blob_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tensorize Image Data using TensorFlow"
    )
    parser.add_argument(
        "-sb",
        "--source_bucket_name",
        default="mini-215-multiclass-car-bucket",
        help="Source GCS bucket name",
    )
    parser.add_argument(
        "-c",
        "--csv_blob_name",
        default="car_preprocessed_folder/class_label.csv",
        help="Blob name of the CSV file in GCS",
    )
    parser.add_argument(
        "-db",
        "--dest_bucket_name",
        default="mini-tensor-bucket",
        help="Destination GCS bucket name",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="./no_ship",
        help="Local directory to save temporary files",
    )
    args = parser.parse_args()

    # Set up local paths
    local_csv_path = os.path.join(args.output_dir, "class_label.csv")
    local_image_dir = os.path.join(args.output_dir, "all_images")
    tfrecord_path = os.path.join(args.output_dir, "data.tfrecord")
    print("start")
    # Step 1: Download data from source bucket
    df = download_data(
        args.source_bucket_name, args.csv_blob_name, local_image_dir, local_csv_path
    )
    print("Step 1 done.")
    # Step 2: Process images and save as TFRecord
    process_and_save_tfrecord(df, local_image_dir, tfrecord_path)
    print("Step 2 done.")
    # Step 3: Upload TFRecord to destination bucket
    upload_to_bucket(args.dest_bucket_name, tfrecord_path)
    print("Step 3 done.")

import os
import io
import argparse
import shutil
import glob
from google.cloud import storage


def download(source_dir, bucket_name, destination_dir):
    print("Downloading")

    shutil.rmtree(source_dir, ignore_errors=True, onerror=None)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(
        prefix=source_dir, match_glob="**.jpg"
    )  # ** means everything include /
    for blob in blobs:
        directory_path = os.path.dirname(blob.name)
        os.makedirs(os.path.join(destination_dir, directory_path), exist_ok=True)
        blob.download_to_filename(os.path.join(destination_dir, blob.name))
        print(f"Downloaded {blob.name}")

    print("Downloaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Data from GCP bucket")

    parser.add_argument("-s", "--source")
    parser.add_argument("-b", "--bucket")
    parser.add_argument("-d", "--destination", default=".")

    args = parser.parse_args()
    bucket_name = args.bucket if args.bucket else "mini-215-multiclass-car-bucket"
    download(args.source, bucket_name, args.destination)
    print("done")

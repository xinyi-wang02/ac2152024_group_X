from image_train_preparation.tensorizing import download_data
from conftest import MINI_BUCKET_NAME, count_files
import os


def test_download_data():
    csv_blob_name = "car_preprocessed_folder/class_label.csv"
    output_dir = "no_ship/tensor_output_1"
    local_csv_path = os.path.join(output_dir, "class_label.csv")
    local_image_dir = os.path.join(output_dir, "all_images")
    os.makedirs(output_dir, exist_ok=True)
    download_data(MINI_BUCKET_NAME, csv_blob_name, local_image_dir, local_csv_path)
    assert count_files(local_image_dir, ".jpg") > 0
    assert count_files(output_dir, ".csv") == 1

import sys, os

# Make sure that the application source directory (this directory's parent) is
# on sys.path.

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, here)

TEST_BUCKET_NAME = "car_class_test_bucket"
MINI_BUCKET_NAME = "mini-215-multiclass-car-bucket"


def delete_ds_store_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

def count_files(folder, file_type):
    file_count = 0
    for root, _, files in os.walk(folder):
        file_count += sum(1 for file in files if file.lower().endswith(file_type))
    return file_count
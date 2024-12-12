#### Test Container and Documentation (tests)

- This container runs a series of tests to verify data preprocessing, tensorizing, model training, model deployment, and one API service's functionalities. PyTest is used to ensure accurate data upload, processing, tensorization from cloud storage, and to test model training and deployment functionalities as much as possible.
- The coverage report indicates an 82% coverage rate.

Detailed Report on Uncovered Parts

- **`src/data_preprocess/data_loader.py`**
  Lines 51-61: Argparse arguments.

- **`src/data_preprocess/preprocess.py`**
  Line 64: Requires >1000 images to cover, but a mini dataset was used for testing.
  Lines 72-73: Coverage requires handling all edge cases. Attempts to cover this using the function `test_process_images_exception_handling` in `src/tests/test_data_preprocess.py` did not trigger coverage.
  Lines 77-114: Argparse arguments.

- **`src/data_preprocess_mini/data_loader.py`**
  Lines 51-61: Argparse arguments.

- **`src/data_preprocess_mini/download.py`**
  Lines 26-35: Argparse arguments.

- **`src/data_preprocess_mini/preprocess.py`**
  Line 69: Requires >1000 images to cover, but a mini dataset was used for testing.
  Lines 77-78: Coverage requires handling all edge cases. Attempts to cover this using the function `test_process_images_exception_handling` in `src/tests/test_data_preprocess_mini.py` did not trigger coverage.
  Lines 82-119: Argparse arguments.

- **`src/image_train_preparation/tensorizing.py`**
  Lines 96-140: Argparse arguments.

- **`src/image_train_preparation_20k/tensorizing.py`**
  Line 45: Requires >100 images to cover, but a mini dataset was used for testing.
  Lines 101-159: Argparse arguments.

- **`src/model_training/model_training_v3.py`**
  Lines 87-113: Coverage requires outputs from other functions, which are difficult to simulate in the test environment.
  Lines 118-175: Argparse arguments.
  Lines 179-217: Argparse arguments.

- **Test Scripts**
  Files such as `conftest.py`, `test_api_dict.py`, ... , `test_model_deploy.py`, and `test_model_training.py` themselves are not covered in the report.

### Untested Scripts

- **`src/api_service/dictionary.py`**
  Tested with `test_api_dict.py` but not included in the coverage report because the tested script is not functionalized and is not imported into the test script.

- **`src/api_service/server.py`**
  Not required to test.

- **`src/deployment/`**
  No `.py` files to test.

- **`src/frontend/`**
  No `.py` files to test.

- **`src/model_deployment/`**
  Tested with `test_model_deploy.py` but not included in the coverage report because the tested script is not functionalized and is not imported into the test script.

- **`src/workflow/`**
  Not tested as the script's logic is simple (running all Docker images for model deployment on Vertex AI), and a similar procedure is repeated using Ansible in `src/deployment/`.

Instructions to Run Tests Manually

enter the below commands in your local terminal:
-   cd ~/src/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

if the user would like to test a single function, type the following after container is running:
-   ./run_single_test.sh

if the user would like to test all function, type the following after container is running:
-   ./run_test.sh

The user could change the function that they would like to test in `run_single_test.sh`

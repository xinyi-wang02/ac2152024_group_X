#### Mini Preprocess container (data_preprocess_mini)

This folder follows the exact same structure as the previous one: `src/data_preprocess`

There is 1 additional script compared with `src/data-preprocess`

(1)`src/data_preprocess_mini/download.py` - Here we download content in our GCP Bucket to local before tensorizing the images.

To run Dockerfile - enter the below commands in your local terminal:

-   cd ~/src/data-preprocess/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

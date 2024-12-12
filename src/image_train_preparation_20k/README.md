#### Tensorizing container (image_train-preparation_20k)

This folder follows the exact same structure as the previous one: `src/image_train_preparation`

The major modification is in the script `src/image_train_preparation_20k/tensorizing.py`

(1)`src/image_train_preparation_20k/tensorizing.py` - Here we download a randomly sampled set of 20,000 images from the augmented pool of approximately 90,000 images and tensorize them. As advised by our teaching fellow, Javier, the complete random sampling method poses a high risk of class imbalance in the dataset used for model training, which needs to be addressed through stratified sampling or downsampling in future training iterations.

To run Dockerfile - enter the below commands in your local terminal:

-   cd ~/src/image_train_preparation/
-   chmod +x docker-shell.sh
-   ./docker-shell.sh

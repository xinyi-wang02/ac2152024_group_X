### IMPORT LIBRARIES
import os
import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAvgPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import InceptionV3
from google.cloud import storage
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.login()

### SETTING CONSTANT VARIABLES
SECRETS_PATH = "/content/data-service-account-model.json"
BUCKET_NAME = "tensor-bucket-20k"
LOCAL_PATH = "/content/tensor"
LOCAL_TFRECORD_PATH = "/content/tensor/data.tfrecord"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 50
WNB_PROJECT_NAME = "model_train_inception_215_group"


### DEFINING FUNCTIONS FOR MODEL TRAIN
def download_tensorized_data_from_bucket(secrets_path, bucket_name, local_dir):
    client = storage.Client.from_service_account_json(secrets_path)
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    os.makedirs(local_dir, exist_ok=True)

    for blob in blobs:
        local_path = os.path.join(local_dir, blob.name)
        local_dir_path = os.path.dirname(local_path)
        os.makedirs(local_dir_path, exist_ok=True)

        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")


feature_description = {
    "image_raw": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


def parse_tfrecord(LOCAL_TFRECORD_PATH):
    # Parse the tensorized data using the feature description
    parsed_example = tf.io.parse_single_example(
        LOCAL_TFRECORD_PATH, feature_description
    )

    # Decode the raw bytes to get the image
    image = tf.io.decode_raw(parsed_example["image_raw"], tf.uint8)
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

    # Get the label
    label = parsed_example["label"]
    return image, label


# One-hot encode labels if using categorical crossentropy
def one_hot_encode(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


### PREPARATION FOR MODEL TRAINING
# download tensorized data from bucket and read in as dataset
download_tensorized_data_from_bucket(SECRETS_PATH, BUCKET_NAME, LOCAL_PATH)
raw_dataset = tf.data.TFRecordDataset(LOCAL_TFRECORD_PATH)

# Parse the tensorized data and reshape the images to prepare for training
parsed_dataset = raw_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

# Determine the number of classes
labels = []
for _, label in parsed_dataset:
    labels.append(label.numpy())
unique_labels = set(labels)
NUM_CLASSES = len(unique_labels)
print(f"Number of classes: {NUM_CLASSES}")

# one-hot encode the classes
parsed_dataset = parsed_dataset.map(one_hot_encode, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and split the dataset into training and validation sets
dataset = parsed_dataset.shuffle(buffer_size=10000)
dataset_size = len(labels)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Batch and prefetch the datasets
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)


### START MODEL TRAINING
# Initialize WandB project
wandb.init(
    project=WNB_PROJECT_NAME,
    config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "architecture": "InceptionV3"},
)

# Model Name
name3 = "CarNetV3"

# Pretrained Model
base_model = InceptionV3(
    include_top=False,
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS),
    weights="imagenet",
)
base_model.trainable = False  # Freeze the Weights

# Model
CarNetV3 = Sequential(
    [
        base_model,
        GlobalAvgPool2D(),
        Dense(224, activation="leaky_relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ],
    name=name3,
)

CarNetV3.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint(
        filepath="best_model.keras", monitor="val_loss", save_best_only=True, verbose=1
    ),
    WandbMetricsLogger(),
    WandbModelCheckpoint("inception_models_215.keras"),
]

# Train
start_time = time.time()

history = CarNetV3.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)
execution_time = (time.time() - start_time) / 60.0
print(f"Training completed in {execution_time:.2f} minutes")

# Evaluate CarNetV3
val_loss, val_accuracy = CarNetV3.evaluate(val_dataset)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

wandb.finish()

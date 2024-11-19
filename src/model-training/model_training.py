import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import ResNet50
from google.cloud import storage
import wandb
from wandb.integration.keras import WandbCallback

# Constants
TFRECORD_BUCKET_NAME = "mini-tensor-bucket"
TFRECORD_FILE_NAME = "data.tfrecord"
LOCAL_TFRECORD_PATH = "no_ship/data.tfrecord"
NUM_CLASSES = None 
BATCH_SIZE = 32
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3
EPOCHS = 3 
BUCKET_URI = "gs://mini_model_wnb/test_model"
# Function to download the TFRecord file from GCS
def download_tfrecord(bucket_name, blob_name, local_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded TFRecord file to {local_path}")

# Define the feature description for parsing the TFRecord
feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

# Function to parse TFRecord examples
def parse_tfrecord_example(example_proto):
    # Parse the input tf.train.Example proto using the feature description
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode the raw bytes to get the image
    image = tf.io.decode_raw(parsed_example['image_raw'], tf.uint8)
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    
    # Get the label
    label = parsed_example['label']
    return image, label

# Download the TFRecord file from GCS
download_tfrecord(TFRECORD_BUCKET_NAME, TFRECORD_FILE_NAME, LOCAL_TFRECORD_PATH)

# Create a TensorFlow Dataset from the TFRecord file
raw_dataset = tf.data.TFRecordDataset(LOCAL_TFRECORD_PATH)

# Parse the dataset
parsed_dataset = raw_dataset.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)

# Determine the number of classes
labels = []
for _, label in parsed_dataset:
    labels.append(label.numpy())
unique_labels = set(labels)
NUM_CLASSES = len(unique_labels)
print(f"Number of classes: {NUM_CLASSES}")

# One-hot encode labels if using categorical crossentropy
def one_hot_encode(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

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

# Define the model using transfer learning (e.g., ResNet50)
base_model = ResNet50(include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')  # Use 'sigmoid' for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Use 'binary_crossentropy' for binary classification
    metrics=['accuracy']
)

# Set up callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
]

# Train the model
start_time = time.time()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
execution_time = (time.time() - start_time) / 60.0
print(f"Training completed in {execution_time:.2f} minutes")

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Preprocess Image for inference
def preprocess_image(bytes_input):
    decoded = tf.io.decode_jpeg(bytes_input, channels=3)
    decoded = tf.image.convert_image_dtype(decoded, tf.float32)
    resized = tf.image.resize(decoded, size=(224, 224))
    return resized

@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def preprocess_function(bytes_inputs):
    decoded_images = tf.map_fn(
        preprocess_image, bytes_inputs, dtype=tf.float32, back_prop=False
    )
    return {"model_input": decoded_images}

@tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
def serving_function(bytes_inputs):
    images = preprocess_function(bytes_inputs)
    results = model_call(**images)
    return results

model_call = tf.function(model.call).get_concrete_function(
    [
        tf.TensorSpec(
            shape=[None, 224, 224, 3], dtype=tf.float32, name="model_input"
        )
    ]
)

tf.saved_model.save(
            model,
            BUCKET_URI,
            signatures={"serving_default": serving_function},
        )

print("Model Uploading Finished")
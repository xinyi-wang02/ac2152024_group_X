import tensorflow as tf
import os
import wandb
from google.cloud import storage
from model_training.model_training_v3 import (
    parse_tfrecord,
    one_hot_encode,
    initialize_wandb,
    build_model,
    train_model,
    evaluate_model,
)


def test_parse_tfrecord():
    """Test parsing a TFRecord to ensure the function can handle actual TFRecord files."""
    #tfrecord_path = "gs://mini-tensor-bucket/data.tfrecord"
    local_tfrecord_path = "no_ship/test_data.tfrecord"

    # Download the TFRecord from the bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket("mini-tensor-bucket")
    blob = bucket.blob("data.tfrecord")
    blob.download_to_filename(local_tfrecord_path)

    # Read and parse TFRecord
    raw_dataset = tf.data.TFRecordDataset(local_tfrecord_path)
    for raw_record in raw_dataset.take(1):
        image, label = parse_tfrecord(raw_record, 224, 224, 3)
        assert image.shape == (
            224,
            224,
            3,
        ), "Parsed image does not have the expected shape"
        assert label.dtype == tf.int64, "Parsed label is not of type int64"

    os.remove(local_tfrecord_path)


def test_one_hot_encode():
    image = tf.ones((224, 224, 3), tf.float32)
    label = 3
    num_classes = 5

    encoded_image, encoded_label = one_hot_encode(image, label, num_classes)

    assert encoded_image.shape == (224, 224, 3)
    assert encoded_label.shape == (num_classes,)
    assert encoded_label.numpy()[label] == 1


def test_initialize_wandb():
    """Test that WandB is initialized properly with the correct configuration."""
    project_name = "test_project"
    epochs = 1
    batch_size = 2
    architecture = "InceptionV3"
    wandb.login()

    try:
        initialize_wandb(project_name, epochs, batch_size, architecture)
        assert wandb.run is not None, "WandB run was not properly initialized"
        assert (
            wandb.run.config["epochs"] == epochs
        ), "WandB epochs config not set correctly"
        assert (
            wandb.run.config["batch_size"] == batch_size
        ), "WandB batch_size config not set correctly"
        assert (
            wandb.run.config["architecture"] == architecture
        ), "WandB architecture config not set correctly"
    finally:
        wandb.finish()


def test_build_model():
    input_shape = (224, 224, 3)
    num_classes = 5
    model = build_model(input_shape, num_classes)

    assert model.input_shape[1:] == input_shape
    assert model.output_shape[-1] == num_classes


def test_train_model():
    # Mock model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Mock dataset
    x = tf.random.uniform((100, 28, 28, 1))
    y = tf.one_hot(tf.random.uniform((100,), maxval=10, dtype=tf.int32), 10)
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)
    val_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)

    # Train
    history = train_model(model, train_dataset, val_dataset, epochs=1, callbacks=[])

    assert history.history["loss"] is not None


def test_evaluate_model():
    """Test the evaluate_model function using a simple mock model and dataset."""
    # Mock model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Mock dataset
    x = tf.random.uniform((100, 28, 28, 1))
    y = tf.one_hot(tf.random.uniform((100,), maxval=10, dtype=tf.int32), 10)
    val_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)

    # Evaluate
    val_loss, val_accuracy = evaluate_model(model, val_dataset)
    assert val_loss is not None, "Validation loss should not be None"
    assert val_accuracy is not None, "Validation accuracy should not be None"
    assert val_loss >= 0, "Validation loss should be non-negative"
    assert 0 <= val_accuracy <= 1, "Validation accuracy should be between 0 and 1"


# def test_upload_model_to_gcs():
# """Test the upload_model_to_gcs function by creating a dummy model and uploading it to GCS."""
# model_bucket_uri = "gs://mini_model_wnb/mock_model"
# local_model_dir = "no_ship/mock_model"
# dummy_jpeg_path = "resources/upload_test_images/test_images/00128.jpg"

# Load a valid JPEG into a tensor
# with open(dummy_jpeg_path, 'rb') as f:
# jpeg_bytes = f.read()
# jpeg_tensor = tf.constant([jpeg_bytes], dtype=tf.string)

# Create a simple model
# model = tf.keras.Sequential([
# tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
# tf.keras.layers.Dense(128, activation='relu'),
# tf.keras.layers.Dense(10, activation='softmax')
# ])

# Create a concrete function for model call
# model_call = tf.function(model.call).get_concrete_function(
# tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)
# )

# @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
# def preprocess_image_warmup(bytes_inputs):
# def process_single_image(x):
# decoded = tf.io.decode_jpeg(x, channels=3)
# decoded = tf.image.convert_image_dtype(decoded, tf.float32)
# resized = tf.image.resize(decoded, size=(224, 224))
# return resized

# Use tf.map_fn to apply process_single_image to each element in the batch
# return tf.map_fn(process_single_image, bytes_inputs, dtype=tf.float32, back_prop=False)

# _ = preprocess_image_warmup(jpeg_tensor)

# if os.path.exists(local_model_dir):
# shutil.rmtree(local_model_dir)
# os.makedirs(local_model_dir, exist_ok=True)

# try:
# Upload the model to GCS
# upload_model_to_gcs(model, model_call, model_bucket_uri)

# Check that the model files exist on GCS
# storage_client = storage.Client()
# gcs_filesystem = gcsfs.GCSFileSystem()
# uploaded_files = gcs_filesystem.ls(model_bucket_uri)

# assert len(uploaded_files) > 0, "No files were uploaded to the GCS bucket"
# assert any("saved_model.pb" in f for f in uploaded_files), "Saved model file (saved_model.pb) not found in GCS"
# finally:
# Cleanup local directory
# if os.path.exists(local_model_dir):
# shutil.rmtree(local_model_dir)

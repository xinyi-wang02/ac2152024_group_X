import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAvgPool2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import InceptionV3
import gcsfs
import tempfile
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import argparse


feature_description = {
    "image_raw": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


def parse_tfrecord(example_proto, image_height, image_width, num_channels):
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_raw(parsed_example["image_raw"], tf.uint8)
    image = tf.reshape(image, [image_height, image_width, num_channels])
    image = tf.cast(image, tf.float32) / 255.0
    label = parsed_example["label"]
    return image, label


def one_hot_encode(image, label, num_classes):
    label = tf.one_hot(label, num_classes)
    return image, label


def initialize_wandb(project_name, epochs, batch_size, architecture):
    wandb.init(
        project=project_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "architecture": architecture,
        },
    )


def build_model(input_shape, num_classes, model_name="CarNetV3"):
    base_model = InceptionV3(
        include_top=False, input_shape=input_shape, weights="imagenet"
    )
    base_model.trainable = False
    model = Sequential(
        [
            base_model,
            GlobalAvgPool2D(),
            Dense(224, activation="leaky_relu"),
            Dense(num_classes, activation="softmax"),
        ],
        name=model_name,
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model(model, train_dataset, val_dataset, epochs, callbacks):
    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    execution_time = (time.time() - start_time) / 60.0
    print(f"Training completed in {execution_time:.2f} minutes")
    return history


def evaluate_model(model, val_dataset):
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")
    return val_loss, val_accuracy


def upload_model_to_gcs(model, model_call, model_bucket_uri):
    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
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

    gcs_file_system = gcsfs.GCSFileSystem()
    with tempfile.TemporaryDirectory() as temp_dir:
        tf.saved_model.save(
            model, temp_dir, signatures={"serving_default": serving_function}
        )
        gcs_file_system.put(temp_dir, model_bucket_uri, recursive=True)
    print("Model Uploading Finished")


def main(args):
    # Load dataset from GCS Bucket
    raw_dataset = tf.data.TFRecordDataset(args.tfrecord_gcs_path)
    parsed_dataset = raw_dataset.map(
        lambda x: parse_tfrecord(
            x, args.image_height, args.image_width, args.num_channels
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Determine the number of classes
    labels = [label.numpy() for _, label in parsed_dataset]
    num_classes = len(set(labels))
    print(f"Number of classes: {num_classes}")

    # One-hot encode labels
    parsed_dataset = parsed_dataset.map(
        lambda image, label: one_hot_encode(image, label, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Prepare datasets
    dataset = parsed_dataset.shuffle(buffer_size=10000)
    train_size = int(0.8 * len(labels))
    train_dataset = (
        dataset.take(train_size).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        dataset.skip(train_size).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    )

    # Initialize WandB
    initialize_wandb(args.project_name, args.epochs, args.batch_size, "InceptionV3")

    # Build, train, and evaluate the model
    input_shape = (args.image_height, args.image_width, args.num_channels)
    model = build_model(input_shape, num_classes)
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=5, verbose=1, restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath="inception_models_v3.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        WandbMetricsLogger(),
        WandbModelCheckpoint("inception_models_v3.h5"),
    ]
    train_model(model, train_dataset, val_dataset, args.epochs, callbacks)
    evaluate_model(model, val_dataset)

    # Upload model to GCS
    model_call = tf.function(model.call).get_concrete_function([
        tf.TensorSpec([None, args.image_height, args.image_width, args.num_channels])
    ])
    upload_model_to_gcs(model, model_call, args.model_bucket_uri)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a car model classification model."
    )
    parser.add_argument(
        "--tfrecord_gcs_path",
        type=str,
        required=True,
        help="Path to the TFRecord file in GCS Bucket.",
    )
    parser.add_argument(
        "--image_height", type=int, default=224, help="Height of the input images."
    )
    parser.add_argument(
        "--image_width", type=int, default=224, help="Width of the input images."
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=3,
        help="Number of color channels in the input images.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--project_name", type=str, required=True, help="Name of the WandB project."
    )
    parser.add_argument(
        "--model_bucket_uri",
        type=str,
        required=True,
        help="URI of the GCS bucket for saving the model.",
    )
    args = parser.parse_args()

    main(args)

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse
from PIL import Image, ImageOps, ImageEnhance


# TODO: function for image resizing and augmentation
def preprocess_image(image):
    processed_images = []
    target_size = (224, 224)  # Message for Harper: You can adjust the size as needed
    # Resize the image
    image_resized = image.resize(target_size)
    processed_images.append((image_resized, ""))

    # Flip horizontally
    image_flipped = ImageOps.mirror(image_resized)
    processed_images.append((image_flipped, "_flip"))

    # Rotate images
    image_rotated1 = image_resized.rotate(-15)
    image_rotated2 = image_resized.rotate(15)
    processed_images.append((image_rotated1, "_rot-15"))
    processed_images.append((image_rotated2, "_rot15"))

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image_resized)
    image_bright = enhancer.enhance(1.5)
    image_dark = enhancer.enhance(0.5)
    processed_images.append((image_bright, "_bright"))
    processed_images.append((image_dark, "_dark"))

    return processed_images


# Function to copy images and record labels
def process_images(data_dir, image_output_dir, output_prefix, image_names):
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    try:
                        # Open the image
                        with Image.open(img_path) as img:
                            # Apply preprocessing and augmentation
                            processed_images = preprocess_image(img)
                            for processed_img, suffix in processed_images:
                                # Generate a unique image name
                                output_image_name = img_name
                                if suffix:
                                    name, ext = os.path.splitext(img_name)
                                    output_image_name = f"{name}{suffix}{ext}"

                                while output_image_name in image_names:
                                    output_image_name = (
                                        output_prefix + output_image_name
                                    )
                                image_names.add(output_image_name)
                                if len(image_names) % 1000 == 0:
                                    print(f"Processed {len(image_names)} files")
                                new_img_path = os.path.join(
                                    image_output_dir, output_image_name
                                )
                                # Save the processed image
                                processed_img.save(new_img_path)
                                # Add the image name and label to the list
                                image_data.append([output_image_name, class_name])
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Data to GCP")

    parser.add_argument("-t", "--train")
    parser.add_argument("-e", "--test")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    train_dir = args.train
    test_dir = args.test
    output_dir = args.output
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "all_images")
    os.makedirs(image_dir, exist_ok=True)
    # List to store the image paths and labels
    image_data = []
    image_names = set()
    # Process both train and test directories
    process_images(train_dir, image_dir, "0", image_names)
    process_images(test_dir, image_dir, "1", image_names)
    print(f"Processed {len(image_names)} files")

    # Create a DataFrame from the collected data
    df = pd.DataFrame(image_data, columns=["image_name", "label"])
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])
    # Save the DataFrame to a CSV file
    label_path = os.path.join(output_dir, "class_label.csv")
    df.to_csv(label_path, index=False)

    label_mapping_df = pd.DataFrame({
        "label": label_encoder.classes_,
        "label_encoded": label_encoder.transform(label_encoder.classes_),
    })
    map_path = os.path.join(output_dir, "class_label_dictionary.csv")
    label_mapping_df.to_csv(map_path, index=False)

    print("done")

import os
import argparse
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from google.cloud import storage

def tensorize_data(bucket_name, csv_blob_name, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Read the CSV file from GCS
    csv_blob = bucket.blob(csv_blob_name)
    csv_data = csv_blob.download_as_string()
    df = pd.read_csv(BytesIO(csv_data))
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize to ImageNet standards
                             std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    labels = []
    
    # Iterate over the dataset
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tensorizing images"):
        image_name = row['image_name']
        label = row['label_encoded']
        image_blob_name = f"all_images/{image_name}"
        image_blob = bucket.blob(image_blob_name)
        
        try:
            # Download the image data as bytes
            image_data = image_blob.download_as_bytes()
            # Open the image from bytes
            img = Image.open(BytesIO(image_data)).convert('RGB')
            # Apply transformations
            img_tensor = transform(img)
            images.append(img_tensor)
            labels.append(label)
        except Exception as e:
            print(f"Error processing image {image_blob_name}: {e}")
    
    # Stack tensors
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Save tensors to files
    images_output_path = os.path.join(output_dir, 'images.pt')
    labels_output_path = os.path.join(output_dir, 'labels.pt')
    torch.save(images_tensor, images_output_path)
    torch.save(labels_tensor, labels_output_path)
    
    print(f"Tensors saved to {output_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorize Image Data from GCS')
    parser.add_argument('-b', '--bucket_name', required=True, help='GCS bucket name')
    parser.add_argument('-c', '--csv_blob_name', required=True, help='Blob name of the CSV file in GCS')
    parser.add_argument('-o', '--output_dir', required=True, help='Directory to save output tensors')
    args = parser.parse_args()
    
    tensorize_data(args.bucket_name, args.csv_blob_name, args.output_dir)

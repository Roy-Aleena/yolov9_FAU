import os
import random

# Paths to the image directories
train_dir = '../datasets/custom_dataset/train/images'  # Update this to your training images directory
val_dir = '../datasets/custom_dataset/val/images'      # Update this to your validation images directory

# Output files
train_file = 'train.txt'
val_file = 'val.txt'

# Function to write image paths to a file
def write_image_paths(image_dir, output_file):
    with open(output_file, 'w') as file:
        for root, _, files in os.walk(image_dir):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if needed
                    file.write(os.path.join(root, filename) + '\n')

# Generate train.txt
write_image_paths(train_dir, train_file)

# Generate val.txt
write_image_paths(val_dir, val_file)

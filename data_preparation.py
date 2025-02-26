# This script is a utility to create multiple samples of an existing image
# so it can be used for batch processing.

import os
import shutil

def create_image_copies(image_path, output_dir, num_copies=50):
    """Creates multiple copies of an image with sequential filenames."""
    
    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, num_copies + 1):
        new_filename = f"{i}.png"
        new_path = os.path.join(output_dir, new_filename)
        shutil.copy(image_path, new_path)
        print(f"Created: {new_path}")

image_path = "resources/image_key.png" # Replace with your own desired path.
output_dir = "resources/dataset" # Replace with your own desired path.

create_image_copies(image_path, output_dir)

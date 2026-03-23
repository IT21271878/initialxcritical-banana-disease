import os
from PIL import Image

def remove_corrupt_images(directory):
    removed_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify() # Verify that it is, in fact, an image
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)
                removed_count += 1
    
    print(f"Removed {removed_count} corrupted images.")

if __name__ == "__main__":
    dataset_dir = r"c:\Code\banana-disease-app\backend\dataset"
    print(f"Scanning for corrupted images in {dataset_dir}...")
    remove_corrupt_images(dataset_dir)

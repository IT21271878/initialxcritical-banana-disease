import os
import shutil
import numpy as np
from PIL import Image

def generate_mock_dataset():
    base_dir = r"c:\Code\banana-disease-app\backend\mock_dataset"
    classes = ["Sigatoka Disease", "Panama Disease", "Healthy", "not detected"]
    
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        
        # Generate 20 random noise images per class
        for i in range(20):
            # Create a random image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(cls_dir, f"mock_{i}.jpg"))
            
    print(f"Generated mock dataset at {base_dir} with {len(classes)} classes.")

if __name__ == "__main__":
    generate_mock_dataset()

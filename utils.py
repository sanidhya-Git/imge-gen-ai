# utils.py
from datetime import datetime
import os

def ensure_output_dir(path="generated_images"):
    os.makedirs(path, exist_ok=True)
    return path

def save_image(image, prefix, count=None, output_dir="generated_images"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{prefix}_{timestamp}"
    if count is not None:
        filename += f"_{count}"
    filename += ".png"
    image.save(filename)
    print(f"✅ Saved: {filename}")

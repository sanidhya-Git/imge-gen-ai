# main.py
from pipeline_loader import load_pipelines
from utils import ensure_output_dir, save_image
from PIL import Image
import os

# Load pipelines
text2img, img2img, device = load_pipelines()
output_dir = ensure_output_dir()

# User prompt
mode = input("Choose mode: (1) Text-to-Image  (2) Image-to-Image [1/2]: ").strip()

if mode == "1":
    prompt = input("Enter your prompt: ")
    num_images = int(input("How many images to generate? "))

    print(f"🎨 Generating {num_images} image(s)...")
    for i in range(num_images):
        result = text2img(prompt=prompt).images[0]
        save_image(result, "text2img", count=i + 1, output_dir=output_dir)

elif mode == "2":
    image_path = input("Enter path to input image: ").strip()
    prompt = input("Enter your prompt: ")

    if not os.path.exists(image_path):
        print("❌ Image not found.")
    else:
        init_image = Image.open(image_path).convert("RGB").resize((1024, 1024))
        result = img2img(prompt=prompt, image=init_image, strength=0.6).images[0]
        save_image(result, "img2img", output_dir=output_dir)

else:
    print("❌ Invalid mode.")

print("🎉 Done!")

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os
from datetime import datetime

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# Load latest SDXL model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# Load Text-to-Image pipeline
text2img_pipeline = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    use_safetensors=True,
    variant="fp16" if device == "cuda" else None
).to(device)

# Load Image-to-Image pipeline
img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    use_safetensors=True,
    variant="fp16" if device == "cuda" else None
).to(device)

# Ensure output directory exists
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Prompt user input
mode = input("Choose mode: (1) Text-to-Image  (2) Image-to-Image [1/2]: ").strip()

if mode == "1":
    prompt = input("Enter your prompt: ")
    num_images = int(input("How many images to generate? "))

    print(f"🎨 Generating {num_images} image(s) from text...")
    for i in range(num_images):
        result = text2img_pipeline(prompt=prompt).images[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/text2img_{timestamp}_{i+1}.png"
        result.save(filename)
        print(f"✅ Saved: {filename}")

elif mode == "2":
    image_path = input("Enter path to input image: ").strip()
    prompt = input("Enter your prompt: ")

    if not os.path.exists(image_path):
        print("❌ Image not found. Check the path.")
    else:
        init_image = Image.open(image_path).convert("RGB").resize((1024, 1024))
        result = img2img_pipeline(prompt=prompt, image=init_image, strength=0.6).images[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/img2img_{timestamp}.png"
        result.save(filename)
        print(f"✅ Image saved: {filename}")
else:
    print("❌ Invalid mode. Please choose 1 or 2.")

print("🎉 Done!")

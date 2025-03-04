import torch
from diffusers import StableDiffusionPipeline
import os
from datetime import datetime

# Disable the safety checker function
def dummy_checker(images, **kwargs):
    return images, False


device = "GPU"
torch_dtype = torch.float32  

pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch_dtype
)

pipeline.safety_checker = dummy_checker  
pipeline.to(device)  

print("🖥️ Running on CPU only (no GPU detected or requested).")

output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)


prompt = input("Enter your prompt: ")
num_images = int(input("How many images to generate? "))

print(f"🔄 Generating {num_images} image(s) for prompt: '{prompt}' on CPU.")


for i in range(num_images):
    image = pipeline(prompt).images[0]


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/image_{timestamp}_{i + 1}.png"
    image.save(filename)
    print(f"✅ Image saved: {filename}")

print("🎉 Image generation completed successfully on CPU!")

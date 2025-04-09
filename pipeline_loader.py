# pipeline_loader.py
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

def try_load_pipeline(device, torch_dtype, variant):
    text2img = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant=variant,
    ).to(device)

    img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant=variant,
    ).to(device)

    return text2img, img2img

def load_pipelines():
    try:
        # Try GPU
        print("⚡ Trying to load pipeline on GPU...")
        device = "cuda"
        torch_dtype = torch.float16
        variant = "fp16"

        text2img, img2img = try_load_pipeline(device, torch_dtype, variant)
        print(" Loaded on GPU")

    except Exception as e:
        # Fallback to CPU
        print("⚠️ GPU not available or failed. Falling back to CPU.")
        print(f"🔍 Error: {e}")

        device = "cpu"
        torch_dtype = torch.float32
        variant = None

        text2img, img2img = try_load_pipeline(device, torch_dtype, variant)
        print(" Loaded on CPU")

    return text2img, img2img, device

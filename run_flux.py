import torch
import os
import base64
import argparse
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from diffusers.utils import load_image
from typing import Tuple
from pathlib import Path
from PIL import Image
import uuid

STORAGE_DIR: Path = Path.home() / "Pictures" 

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def load_flux_img_to_img():
    pipeline = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    return pipeline

def load_flux():
    pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    return pipeline

def generate_image(pipeline, prompt, init_image=None, height=1024, width=1024, guideance_scale=0, num_images_per_prompt=1, num_inference_steps=50):
    images = pipeline(
            prompt=prompt,
            guidance_scale=guideance_scale,
            height=height,
            width=width,
            max_sequence_length=256,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            image=init_image
            ).images
    return images

def generate_random_string(length=16) -> str:
    return str(uuid.uuid4())

def parse_dimensions(dim_str: str) -> Tuple[int, int]:
    try:
        width, height = map(int, dim_str.split(':'))
        return (width, height)
    except ValueError:
        raise argparse.ArgumentError('Dimensions must be in format width:height')

def main():
    try:
        args = parser.parse_args()

        pipeline = load_flux_img_to_img()
        pipeline.to(torch.float16)

        target_img = STORAGE_DIR / "1517481062292.jpg"

        target_img_path = target_img.resolve(strict=True)

        image = Image.open(target_img_path)

        init_image = load_image(image).resize((256, 256))

        width, height = args.size

        images = generate_image(pipeline, init_image=init_image, prompt=args.prompt, width=width, height=height, guideance_scale=args.guideance_scale, num_images_per_prompt=args.number)
        for image in images:
            filename = generate_random_string()
            filepath = STORAGE_DIR / "Flux" / f"{filename}.png"
            image.save(filepath)
    except FileNotFoundError:
        print("\n Target image doesn't exist. Exiting...")
        exit(0)
    except KeyboardInterrupt:
        print('\nExiting early...')
        exit(0)

parser = argparse.ArgumentParser(description="Generate some A.I. images", epilog="All done!")

parser.add_argument("-n", "--number", type=int, default=1, help="the number of images you want to generate") 
parser.add_argument("-o", "--output", type=str, default="image", help="the name of the output image")
parser.add_argument("-p", "--prompt", type=str, required=True, help="the prompt")
parser.add_argument("-gs", "--guideance-scale", type=float, default=0)
parser.add_argument("--size", type=parse_dimensions, default="1024:1024")

if __name__ == "__main__":
    main()

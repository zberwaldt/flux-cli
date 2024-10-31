import torch
import os
import base64
import argparse
import argcomplete
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from diffusers.utils import load_image
from typing import Tuple
from pathlib import Path
from PIL import Image
import uuid
import logging

STORAGE_DIR: Path = Path.home() / "Pictures" 

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("run_flux")

def image_completer(prefix, parsed_args, **kwargs):
    image_dir = STORAGE_DIR / "Flux"
    return [
        filename for filename in os.listdir(image_dir)
        if filename.startswith(prefix) and os.path.isfile(os.path.join(image_dir, filename))
    ]

def record_prompt(prompt, filename="prompts.txt"):
    try:
        with open(filename, "r") as file:
            existing_prompts = set(line.strip() for line in file)
    except FileNotFoundError:
        existing_prompts = set()

    if prompt not in existing_prompts:
        with open(filename, "a") as file:
            file.write(prompt + "\n")
        logger.info(f"Recording new prompt: \"{prompt}\"")
    else:
        logger.info(f"Prompt already exists in the file: \"{prompt}\"")

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

def generate_image(pipeline, prompt, prompt_2=None, init_image=None, height=1024, width=1024, guideance_scale=0, num_images_per_prompt=1, num_inference_steps=50):
    kwargs = { 
          "prompt": prompt,
          "prompt_2": prompt_2,
          "guidance_scale": guideance_scale,
          "height":height,
          "width": width,
          "max_sequence_length": 256,
          "num_inference_steps": num_inference_steps,
          "num_images_per_prompt": num_images_per_prompt
    }

    if isinstance(pipeline, FluxImg2ImgPipeline) and init_image is not None:
        kwargs["image"] = init_image

    images = pipeline(**kwargs).images
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
        logging.basicConfig(filename="flux.log", level=logging.INFO, format='%(asctime)s - %(levelname)s -> %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p")
        
        logger.info("Parsing arguments")

        args = parser.parse_args()

        logger.info("Choosing model...")

        pipeline = load_flux_img_to_img() if args.use_image else load_flux()

        if isinstance(pipeline, FluxPipeline):
            logger.info("Using text-to-image model")
        else:
            logger.info("Using image-to-image model")

        pipeline.to(torch.float16)

        # target_img = STORAGE_DIR / "1517481062292.jpg"
        target_img = STORAGE_DIR / "Flux" / "a23aae99-c8f1-4ce5-b91f-0b732774dadd.png"

        target_img_path = target_img.resolve(strict=True)

        image = Image.open(target_img_path)

        init_image = load_image(image).resize((256, 256))

        width, height = args.size

        record_prompt(args.prompt)

        logger.info(f"Using prompt: \"{args.prompt}\"")

        logger.info("Generating image(s)...")

        images = generate_image(
                pipeline=pipeline, 
                init_image=init_image,
                prompt=args.prompt,
                prompt_2=args.prompt2,
                width=width,
                height=height,
                guideance_scale=args.guideance_scale,
                num_images_per_prompt=args.number
            )

        for image in images:
            filename = generate_random_string()
            filepath = STORAGE_DIR / "Flux" / f"{filename}.png"
            logger.info(f"Saving {filepath}...")
            image.save(filepath)

        logger.info("Finished")
    except FileNotFoundError:
        print("\n Target image doesn't exist. Exiting...")
        exit(0)
    except KeyboardInterrupt:
        print('\nExiting early...')
        exit(0)
    except Exception as e:
        print(f"An error occured: {e}")
        exit(1)

parser = argparse.ArgumentParser(description="Generate some A.I. images", epilog="All done!")

parser.add_argument("-n", "--number", type=int, default=1, help="the number of images you want to generate") 
parser.add_argument("-o", "--output", type=str, default="image", help="the name of the output image")
parser.add_argument("-p", "--prompt", type=str, required=True, help="the prompt")
parser.add_argument("-p2", "--prompt2", type=str, help="A second prompt")
parser.add_argument("-gs", "--guideance-scale", type=float, default=0)
parser.add_argument("--size", type=parse_dimensions, default="1024:1024", help="the size of the output images")
parser.add_argument("-u", "--use-image", action="store_true", help="use a predefined image")

# parser.add_argument("-b", "--base-image").completer = image_completer

# argcomplete.autocomplete(parser)

if __name__ == "__main__":
    main()

import os
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Optional
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

def load_flux():
    import torch
    from diffusers import FluxPipeline

    pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    return pipeline

@dataclass(frozen=True)
class GenerateImageConfig:
    prompt: str
    prompt_2: Optional[str] = None
    init_image: Optional[Image] = None
    strength: int = 0.0
    guidance_scale: float = 0.0
    height: int = 1024
    width: int = 1024
    num_images_per_prompt: int = 1
    num_inference_steps: int = 50

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

def generate_image(pipeline, config: GenerateImageConfig):
    images = pipeline(**config.to_dict()).images
    return images

def generate_random_string(length=16) -> str:
    return str(uuid.uuid4())

def parse_dimensions(dim_str: str) -> Tuple[int, int]:
    try:
        width, height = map(int, dim_str.split(':'))
        return width, height
    except ValueError:
        raise argparse.ArgumentError('Dimensions must be in format width:height')

def main():
    logging.basicConfig(filename="flux.log", level=logging.INFO, format='%(asctime)s - %(levelname)s -> %(message)s',
                        datefmt="%m/%d/%Y %I:%M:%S %p")

    logger.info("Parsing arguments")

    parser = argparse.ArgumentParser(description="Generate some A.I. images", epilog="All done!")

    parser.add_argument("-n", "--number", type=int, default=1, help="the number of images you want to generate")
    parser.add_argument("-o", "--output", type=str, default="image", help="the name of the output image")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="the prompt")
    parser.add_argument("-p2", "--prompt2", type=str, help="A second prompt")
    parser.add_argument("-gs", "--guideance-scale", type=float, default=0)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--size", type=parse_dimensions, default="1024:1024", help="the size of the output images")
    args = parser.parse_args()

    try:
        import torch
        from diffusers.utils import load_image

        logger.info("Choosing model...")

        pipeline = load_flux()

        pipeline.to(torch.float16)

        width, height = args.size

        record_prompt(args.prompt)

        logger.info(f"Using prompt: \"{args.prompt}\"")

        logger.info("Generating image(s)...")

        config = GenerateImageConfig(
            prompt=args.prompt,
            prompt_2=args.prompt2 if args.prompt2 else None,
            width=width,
            height=height,
            strength=args.strength,
            guidance_scale=args.guideance_scale,
            num_images_per_prompt=args.number
        )

        images = generate_image(
                pipeline=pipeline,
                config=config
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


if __name__ == "__main__":
    main()

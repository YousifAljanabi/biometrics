#!/usr/bin/env python3
"""
Pipeline for generating distorted fingerprint images from clean dataset.
Creates 10 distorted variations of each clean image for each distortion type.
"""

import os
import logging
from pathlib import Path

from distorters import (
    radial, stretch, rotate, translate, scale,
    affine_warp, perspective_warp,
    ridge_erosion, partial_loss, elastic,
    add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise,
    compression_loss
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Distortion functions mapping
distortions = {
    "radial": radial,
    "stretch": stretch,
    "rotate": rotate,
    "translate": translate,
    "scale": scale,
    "affine_warp": affine_warp,
    "perspective_warp": perspective_warp,
    "ridge_erosion": ridge_erosion,
    "partial_loss": partial_loss,
    "elastic": elastic,
    "gaussian_noise": add_gaussian_noise,
    "salt_pepper_noise": add_salt_pepper_noise,
    "speckle_noise": add_speckle_noise,
    "compression_loss": compression_loss,
}

def create_distortion_directories(output_root: str) -> None:
    """
    Create directories for each distortion type.

    Args:
        output_root: Root directory for distorted images
    """
    for distortion_name in distortions.keys():
        distortion_dir = os.path.join(output_root, distortion_name)
        os.makedirs(distortion_dir, exist_ok=True)
        logger.info(f"Created directory: {distortion_dir}")

def generate_distorted_images(clean_dir: str, output_root: str, variations_per_image: int = 10) -> None:
    """
    Generate distorted variations of all clean images.

    Args:
        clean_dir: Directory containing clean fingerprint images
        output_root: Root directory for saving distorted images
        variations_per_image: Number of distorted variations to create per image
    """
    # Check if clean directory exists
    if not os.path.exists(clean_dir):
        logger.error(f"Clean directory {clean_dir} does not exist")
        return

    # Create distortion directories
    create_distortion_directories(output_root)

    # Get all image files from clean directory
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    clean_images = []

    for file in os.listdir(clean_dir):
        if Path(file).suffix.lower() in image_extensions:
            clean_images.append(file)

    logger.info(f"Found {len(clean_images)} clean images to process")

    if not clean_images:
        logger.warning("No image files found in clean directory")
        return

    total_images_to_generate = len(clean_images) * len(distortions) * variations_per_image
    logger.info(f"Will generate {total_images_to_generate} distorted images total")

    processed_count = 0
    failed_count = 0

    # Process each clean image
    for clean_image in clean_images:
        clean_path = os.path.join(clean_dir, clean_image)
        image_name = Path(clean_image).stem  # Remove extension

        # Apply each distortion type
        for distortion_name, distortion_func in distortions.items():
            distortion_dir = os.path.join(output_root, distortion_name)

            # Generate multiple variations
            for variation in range(variations_per_image):
                try:
                    # Create output filename with variation number
                    output_filename = f"{distortion_name}_{image_name}_var{variation:02d}.png"
                    output_path = os.path.join(distortion_dir, output_filename)

                    # Apply distortion with random parameters (each function handles its own randomization)
                    distortion_func(clean_path, output_path)

                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count}/{total_images_to_generate} images")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to process {clean_image} with {distortion_name} variation {variation}: {str(e)}")

    logger.info(f"Generation complete: {processed_count} successful, {failed_count} failed")

def main():
    """Main function to run the distortion pipeline."""
    # Configuration using os to get directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    clean_dir = os.path.join(project_root, "dataset", "clean")
    output_root = os.path.join(project_root, "dataset", "distorted")
    variations_per_image = 10

    logger.info(f"Starting distortion pipeline")
    logger.info(f"Current directory: {current_dir}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Clean images directory: {clean_dir}")
    logger.info(f"Output directory: {output_root}")
    logger.info(f"Variations per image: {variations_per_image}")

    generate_distorted_images(clean_dir, output_root, variations_per_image)

if __name__ == "__main__":
    main()

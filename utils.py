import io
import os
from pathlib import Path
from PIL import Image
from typing import Union


def convert_image_to_png(image_data: bytes) -> bytes:
    """
    Convert any image format to PNG format for TensorFlow compatibility.
    Converts to grayscale since the model expects single-channel input.

    Args:
        image_data: Raw image data in bytes

    Returns:
        PNG image data in bytes (grayscale)

    Raises:
        ValueError: If the image data is invalid or cannot be processed
    """
    try:
        # Open the image from bytes
        image = Image.open(io.BytesIO(image_data))

        if image.mode != 'L':
            if image.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                else:
                    background.paste(image, mask=image.split()[-1])
                image = background

            image = image.convert('L')

        png_buffer = io.BytesIO()
        image.save(png_buffer, format='PNG', optimize=False)  # Disable optimization for TF compatibility
        png_buffer.seek(0)

        return png_buffer.getvalue()

    except Exception as e:
        raise ValueError(f"Failed to convert image to PNG: {str(e)}")


def validate_image_data(image_data: bytes) -> bool:
    """
    Validate if the provided bytes represent a valid image.

    Args:
        image_data: Raw image data in bytes

    Returns:
        True if valid image, False otherwise
    """
    try:
        Image.open(io.BytesIO(image_data))
        return True
    except Exception:
        return False


def get_image_info(image_data: bytes) -> dict:
    """
    Get basic information about an image.

    Args:
        image_data: Raw image data in bytes

    Returns:
        Dictionary containing image info (format, size, mode)

    Raises:
        ValueError: If the image data is invalid
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        return {
            'format': image.format,
            'size': image.size,  # (width, height)
            'mode': image.mode,
            'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
        }
    except Exception as e:
        raise ValueError(f"Failed to get image info: {str(e)}")


def write_bytes_to_png(png_bytes: bytes, file_path: str) -> None:
    """
    Write PNG image bytes to a file.

    Args:
        png_bytes: PNG image data in bytes
        file_path: Path where to save the PNG file

    Raises:
        OSError: If unable to write to the specified path
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'wb') as f:
            f.write(png_bytes)
    except Exception as e:
        raise OSError(f"Failed to write PNG to {file_path}: {str(e)}")

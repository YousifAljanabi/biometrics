import sys
from pathlib import Path

# Add parent directory to path to import model_service
sys.path.append(str(Path(__file__).parent.parent))

import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

from training.reconstruction.model_service import get_service
from training.classification.model_service import get_classification_service
from utils import convert_image_to_png, validate_image_data, get_image_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fingerprint Reconstruction API",
    description="API for reconstructing distorted fingerprint images using U-Net with STN",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ReconstructionResponse(BaseModel):
    success: bool
    message: str
    reconstructed_image: Optional[str] = None
    distortion_type: Optional[str] = None
    original_shape: Optional[List[int]] = None
    reconstructed_shape: Optional[List[int]] = None

class HealthResponse(BaseModel):
    status: str
    reconstruction_model_loaded: bool
    classification_model_loaded: bool
    available_distortions: List[str]
    available_classes: List[str]

class DistortionTypesResponse(BaseModel):
    distortion_types: List[str]

class ClassificationResponse(BaseModel):
    success: bool
    message: str
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    top_predictions: Optional[List[dict]] = None
    original_shape: Optional[List[int]] = None

# Global service instances will be initialized on startup
service = get_service()
classification_service = get_classification_service()

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        return HealthResponse(
            status="healthy",
            reconstruction_model_loaded=service.is_model_loaded(),
            classification_model_loaded=classification_service.is_model_loaded(),
            available_distortions=service.get_available_distortions(),
            available_classes=classification_service.get_available_classes()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/distortion-types", response_model=DistortionTypesResponse)
async def get_distortion_types():
    """Get available distortion types."""
    try:
        return DistortionTypesResponse(
            distortion_types=service.get_available_distortions()
        )
    except Exception as e:
        logger.error(f"Error getting distortion types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reconstruct", response_model=ReconstructionResponse)
async def reconstruct_fingerprint(
    file: UploadFile = File(...),
    distortion_type: str = Form(...)
):
    """
    Reconstruct a distorted fingerprint image.

    Args:
        file: Uploaded image file (any format - will be converted to grayscale PNG)
        distortion_type: Type of distortion affecting the image

    Returns:
        Reconstructed image as base64 encoded string
    """
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected image file."
            )

        # Validate distortion type
        available_distortions = service.get_available_distortions()
        if distortion_type not in available_distortions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid distortion type: {distortion_type}. "
                       f"Available types: {available_distortions}"
            )

        # Read image data
        image_data = await file.read()
        logger.info(f"Processing {file.filename} with distortion type: {distortion_type}")

        # Validate image data
        if not validate_image_data(image_data):
            raise HTTPException(
                status_code=400,
                detail="Invalid image data. Please upload a valid image file."
            )

        # Get original image info for reference
        original_info = get_image_info(image_data)
        original_shape = list(original_info['size'])  # (width, height)
        logger.info(f"Original image format: {original_info['format']}, size: {original_info['size']}, mode: {original_info['mode']}")

        # Convert any image format to PNG for consistent processing
        png_data = convert_image_to_png(image_data)
        logger.info("Image converted to PNG format")

        # Reconstruct image using PNG data
        reconstructed = service.reconstruct(png_data, distortion_type)

        # Convert numpy array back to image
        # reconstructed is in range [0, 1], convert to [0, 255]
        reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)

        # Create PIL image
        reconstructed_img = Image.fromarray(reconstructed_uint8, mode='L')

        # Convert to base64 for response
        img_buffer = io.BytesIO()
        reconstructed_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return ReconstructionResponse(
            success=True,
            message="Image reconstructed successfully",
            reconstructed_image=img_base64,
            distortion_type=distortion_type,
            original_shape=original_shape,
            reconstructed_shape=list(reconstructed.shape)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during reconstruction: {e}")
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")

@app.post("/classify", response_model=ClassificationResponse)
async def classify_fingerprint(
    file: UploadFile = File(...)
):
    """
    Classify the distortion type of a fingerprint image.

    Args:
        file: Uploaded image file (any format - will be converted to grayscale PNG)

    Returns:
        Classification result with predicted distortion type and confidence
    """
    if classification_service is None:
        raise HTTPException(status_code=500, detail="Classification service not initialized")

    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Expected image file."
            )

        # Read image data
        image_data = await file.read()
        logger.info(f"Processing {file.filename} for classification")

        # Validate image data
        if not validate_image_data(image_data):
            raise HTTPException(
                status_code=400,
                detail="Invalid image data. Please upload a valid image file."
            )

        # Get original image info for reference
        original_info = get_image_info(image_data)
        original_shape = list(original_info['size'])  # (width, height)
        logger.info(f"Original image format: {original_info['format']}, size: {original_info['size']}, mode: {original_info['mode']}")

        # Convert any image format to PNG for consistent processing
        png_data = convert_image_to_png(image_data)
        logger.info("Image converted to PNG format")

        # Classify the image
        predicted_class, confidence, top_predictions = classification_service.classify(png_data)

        # Format top predictions for response
        top_predictions_formatted = [
            {"class": class_name, "confidence": conf}
            for class_name, conf in top_predictions
        ]

        return ClassificationResponse(
            success=True,
            message="Image classified successfully",
            predicted_class=predicted_class,
            confidence=confidence,
            top_predictions=top_predictions_formatted,
            original_shape=original_shape
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# @app.post("/reconstruct-base64", response_model=ReconstructionResponse)
# async def reconstruct_fingerprint_base64(
#     image_base64: str = Form(...),
#     distortion_type: str = Form(...)
# ):
#     """
#     Reconstruct a distorted fingerprint image from base64 encoded data.
#
#     Args:
#         image_base64: Base64 encoded image data
#         distortion_type: Type of distortion affecting the image
#
#     Returns:
#         Reconstructed image as base64 encoded string
#     """
#     if service is None:
#         raise HTTPException(status_code=500, detail="Service not initialized")
#
#     try:
#         # Decode base64 image
#         try:
#             image_data = base64.b64decode(image_base64)
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
#
#         # Validate distortion type
#         available_distortions = service.get_available_distortions()
#         if distortion_type not in available_distortions:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid distortion type: {distortion_type}. "
#                        f"Available types: {available_distortions}"
#             )
#
#         logger.info(f"Processing base64 image with distortion type: {distortion_type}")
#
#         # Get original image shape for reference
#         original_img = Image.open(io.BytesIO(image_data))
#         original_shape = list(original_img.size)  # (width, height)
#
#         # Reconstruct image
#         reconstructed = service.reconstruct(image_data, distortion_type)
#
#         # Convert numpy array back to image
#         reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)
#         reconstructed_img = Image.fromarray(reconstructed_uint8, mode='L')
#
#         # Convert to base64 for response
#         img_buffer = io.BytesIO()
#         reconstructed_img.save(img_buffer, format='PNG')
#         img_buffer.seek(0)
#         img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
#
#         return ReconstructionResponse(
#             success=True,
#             message="Image reconstructed successfully",
#             reconstructed_image=img_base64,
#             distortion_type=distortion_type,
#             original_shape=original_shape,
#             reconstructed_shape=list(reconstructed.shape)
#         )
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error during reconstruction: {e}")
#         raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
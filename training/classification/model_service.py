#!/usr/bin/env python3
"""
Model service for fingerprint distortion classification.
Provides classification functionality for the API.
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FingerprintClassificationService:
    """Service for loading and using the fingerprint distortion classification model."""

    def __init__(self, model_path: str = "distortion_classifier_best.h5"):
        """
        Initialize the classification service.

        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.input_shape = (256, 256, 1)

    def load_model(self) -> None:
        """Load the trained model and class names."""
        try:
            # Get the directory of this file
            current_dir = Path(__file__).parent

            # Load class names
            class_names_file = current_dir / "class_names.json"
            if class_names_file.exists():
                with open(class_names_file, 'r') as f:
                    self.class_names = json.load(f)
                logger.info(f"Loaded {len(self.class_names)} class names: {self.class_names}")
            else:
                # Fallback to default distortion types
                self.class_names = [
                    "affine_warp", "compression_loss", "elastic", "gaussian_noise",
                    "partial_loss", "perspective_warp", "radial", "ridge_erosion",
                    "rotate", "salt_pepper_noise", "scale", "speckle_noise",
                    "stretch", "translate"
                ]
                logger.warning(f"Class names file not found, using default: {self.class_names}")

            # Load model
            model_file = current_dir / self.model_path
            if model_file.exists():
                logger.info(f"Loading saved model from {model_file}")
                self.model = tf.keras.models.load_model(str(model_file))
                logger.info("Classification model loaded successfully")
            else:
                logger.error(f"Model file {model_file} not found")
                raise FileNotFoundError(f"Model file {model_file} not found")

        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            raise

    def preprocess_image(self, image_data: bytes) -> tf.Tensor:
        """
        Preprocess image data for model input.

        Args:
            image_data: Raw image bytes

        Returns:
            Preprocessed image tensor
        """
        try:
            # Decode image
            img = tf.io.decode_image(image_data, channels=1, expand_animations=False)

            # Convert to float32 and normalize to [0, 1]
            img = tf.image.convert_image_dtype(img, tf.float32)

            # Resize to model input size
            img = tf.image.resize(img, self.input_shape[:2], method="bilinear")

            # Add batch dimension
            img = tf.expand_dims(img, 0)

            return img

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Invalid image data: {e}")

    def classify(self, image_data: bytes) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify the distortion type of a fingerprint image.

        Args:
            image_data: Raw image bytes

        Returns:
            Tuple containing:
            - predicted_class: Most likely distortion type
            - confidence: Confidence score of the prediction
            - top_predictions: List of (class_name, probability) tuples for top 3 predictions
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_data)

            # Run inference
            logger.info("Running classification inference")
            predictions = self.model.predict(img_tensor, verbose=0)

            # Get probabilities
            probabilities = predictions[0]

            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_predictions = [
                (self.class_names[idx], float(probabilities[idx]))
                for idx in top_indices
            ]

            # Get the most likely prediction
            predicted_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            logger.info(f"Classification result: {predicted_class} (confidence: {confidence:.3f})")

            return predicted_class, confidence, top_predictions

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            raise

    def get_available_classes(self) -> List[str]:
        """Get list of available distortion classes."""
        return self.class_names.copy()

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Global service instance
_classification_service_instance = None

def get_classification_service() -> FingerprintClassificationService:
    """Get or create the global classification service instance."""
    global _classification_service_instance
    if _classification_service_instance is None:
        _classification_service_instance = FingerprintClassificationService()
        _classification_service_instance.load_model()
    return _classification_service_instance
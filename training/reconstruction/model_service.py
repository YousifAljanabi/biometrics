import os
import numpy as np
import tensorflow as tf
import logging

from training.reconstruction.model import build_unet
from training.reconstruction.utils import (DISTORTION_CLASSES)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FingerprintReconstructionService:
    """Service for loading and using the U-Net fingerprint reconstruction model."""

    def __init__(self, model_path: str = "elastic_unet_conditional.h5"):
        """
        Initialize the reconstruction service.

        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.input_shape = (256, 256, 1)
        self.num_classes = len(DISTORTION_CLASSES)
        self.distortion_classes = DISTORTION_CLASSES

    def load_model(self) -> None:
        """Load the trained model."""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading saved model from {self.model_path}")
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects={'ssim_l1_loss': self._ssim_l1_loss}
                )
            else:
                logger.warning(f"Model file {self.model_path} not found. Building fresh model.")
                self.model = build_unet(
                    input_shape=self.input_shape,
                    num_cond_classes=self.num_classes,
                    residual=True,
                    use_stn=True
                )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _ssim_l1_loss(self, y_true, y_pred, alpha=0.84):
        """Custom loss function for model loading."""
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        ssim_loss = (1.0 - ssim) / 2.0
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1,2,3])
        return alpha * ssim_loss + (1.0 - alpha) * l1

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

    def get_condition_vector(self, distortion_type: str) -> tf.Tensor:
        """
        Get one-hot condition vector for distortion type.

        Args:
            distortion_type: Type of distortion (must be in DISTORTION_CLASSES)

        Returns:
            One-hot encoded condition vector as TensorFlow tensor
        """
        if distortion_type not in self.distortion_classes:
            raise ValueError(f"Unknown distortion type: {distortion_type}. "
                           f"Available types: {self.distortion_classes}")

        condition = np.zeros((1, self.num_classes), dtype=np.float32)
        condition[0, self.distortion_classes.index(distortion_type)] = 1.0

        # Convert to TensorFlow tensor to match the image tensor type
        return tf.constant(condition)

    def reconstruct(self, image_data: bytes, distortion_type: str) -> np.ndarray:
        """
        Reconstruct a distorted fingerprint image.

        Args:
            image_data: Raw image bytes
            distortion_type: Type of distortion affecting the image

        Returns:
            Reconstructed image as numpy array (H, W) in range [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_data)

            # Get condition vector
            condition = self.get_condition_vector(distortion_type)

            # Run inference
            logger.info(f"Running reconstruction for distortion type: {distortion_type}")
            prediction = self.model([img_tensor, condition])

            # Convert back to numpy and remove batch dimension
            result = prediction.numpy()[0, :, :, 0]

            # Ensure values are in [0, 1] range
            result = np.clip(result, 0.0, 1.0)

            return result

        except Exception as e:
            logger.error(f"Error during reconstruction: {e}")
            raise

    def get_available_distortions(self) -> list:
        """Get list of available distortion types."""
        return self.distortion_classes.copy()

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# Global service instance
_service_instance = None

def get_service() -> FingerprintReconstructionService:
    """Get or create the global service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = FingerprintReconstructionService()
        _service_instance.load_model()
    return _service_instance
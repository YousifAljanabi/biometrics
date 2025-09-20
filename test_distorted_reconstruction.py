#!/usr/bin/env python3
"""
Test script for reconstructing distorted fingerprint images using the conditional elastic UNet model.
Tests the model on images from dataset/distorted/ and displays results with plotting.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from training.reconstruction.model_service import FingerprintReconstructionService
from training.reconstruction.utils import DISTORTION_CLASSES


class DistortedImageReconstructionTest:
    """Test class for fingerprint reconstruction on distorted images."""

    def __init__(self, model_path: str = None):
        """Initialize test with model path."""
        if model_path is None:
            model_path = "elastic_unet_conditional.h5"

        self.model_path = model_path
        self.service = FingerprintReconstructionService(model_path)
        self.dataset_path = "dataset/distorted"

    def load_model(self):
        """Load the reconstruction model."""
        print(f"Loading model from: {self.model_path}")
        self.service.load_model()
        print("Model loaded successfully!")

    def get_sample_images(self, distortion_type: str, num_samples: int = 3):
        """Get sample images from a distortion directory."""
        distortion_dir = os.path.join(self.dataset_path, distortion_type)

        if not os.path.exists(distortion_dir):
            raise ValueError(f"Distortion directory not found: {distortion_dir}")

        image_files = [f for f in os.listdir(distortion_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(image_files) == 0:
            raise ValueError(f"No image files found in {distortion_dir}")

        # Select sample images
        selected_files = image_files[:num_samples]
        return [os.path.join(distortion_dir, f) for f in selected_files]

    def load_image_bytes(self, image_path: str):
        """Load image as bytes for processing."""
        with open(image_path, 'rb') as f:
            return f.read()

    def test_reconstruction(self, distortion_type: str, num_samples: int = 3):
        """Test reconstruction on sample images from a distortion type."""
        print(f"\nTesting reconstruction for distortion type: {distortion_type}")

        # Get sample images
        sample_paths = self.get_sample_images(distortion_type, num_samples)
        print(f"Testing on {len(sample_paths)} sample images")

        # Set up plotting
        fig, axes = plt.subplots(2, len(sample_paths), figsize=(4*len(sample_paths), 8))
        if len(sample_paths) == 1:
            axes = axes.reshape(2, 1)

        for idx, image_path in enumerate(sample_paths):
            print(f"Processing: {os.path.basename(image_path)}")

            # Load image
            image_bytes = self.load_image_bytes(image_path)

            # Reconstruct
            try:
                reconstructed = self.service.reconstruct(image_bytes, distortion_type)

                # Load original for display
                original_tensor = self.service.preprocess_image(image_bytes)
                original_np = original_tensor.numpy()[0, :, :, 0]

                # Plot original (top row)
                axes[0, idx].imshow(original_np, cmap='gray')
                axes[0, idx].set_title(f'Original (Distorted)\n{os.path.basename(image_path)}')
                axes[0, idx].axis('off')

                # Plot reconstructed (bottom row)
                axes[1, idx].imshow(reconstructed, cmap='gray')
                axes[1, idx].set_title(f'Reconstructed')
                axes[1, idx].axis('off')

                print(f"  ✓ Reconstructed successfully")
                print(f"  Original shape: {original_np.shape}, range: [{original_np.min():.3f}, {original_np.max():.3f}]")
                print(f"  Reconstructed shape: {reconstructed.shape}, range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

            except Exception as e:
                print(f"  ✗ Reconstruction failed: {e}")

                # Show error in plot
                axes[0, idx].text(0.5, 0.5, 'Error loading\noriginal',
                                ha='center', va='center', transform=axes[0, idx].transAxes)
                axes[0, idx].set_title(f'Error: {os.path.basename(image_path)}')
                axes[0, idx].axis('off')

                axes[1, idx].text(0.5, 0.5, f'Reconstruction\nfailed:\n{str(e)[:50]}...',
                                ha='center', va='center', transform=axes[1, idx].transAxes)
                axes[1, idx].set_title('Reconstruction Failed')
                axes[1, idx].axis('off')

        plt.tight_layout()
        plt.suptitle(f'Fingerprint Reconstruction Test - {distortion_type.upper()}', y=0.98)
        plt.show()

    def test_all_distortions(self, num_samples: int = 2):
        """Test reconstruction on all available distortion types."""
        print("Testing reconstruction on all distortion types...")

        available_distortions = []
        for distortion in DISTORTION_CLASSES:
            distortion_dir = os.path.join(self.dataset_path, distortion)
            if os.path.exists(distortion_dir):
                available_distortions.append(distortion)

        print(f"Found {len(available_distortions)} available distortion types: {available_distortions}")

        for distortion in available_distortions:
            try:
                self.test_reconstruction(distortion, num_samples=num_samples)
            except Exception as e:
                print(f"Failed to test {distortion}: {e}")
                continue

    def test_single_distortion_type(self, distortion_type: str = "elastic", num_samples: int = 3):
        """Test reconstruction on a specific distortion type."""
        if distortion_type not in DISTORTION_CLASSES:
            raise ValueError(f"Unknown distortion type: {distortion_type}. Available: {DISTORTION_CLASSES}")

        self.test_reconstruction(distortion_type, num_samples)


def main():
    """Main function to run the reconstruction test."""
    print("=== Distorted Fingerprint Reconstruction Test ===")

    # Initialize test
    test = DistortedImageReconstructionTest()

    # Load model
    test.load_model()

    # Test on elastic distortion (since model is named elastic_unet_conditional)
    print("\n" + "="*50)
    test.test_single_distortion_type("elastic", num_samples=3)

    # Optionally test other distortions
    print("\n" + "="*50)
    print("Testing additional distortion types...")

    other_distortions = ["stretch", "gaussian_noise", "salt_pepper_noise"]
    for distortion in other_distortions:
        try:
            test.test_single_distortion_type(distortion, num_samples=2)
        except Exception as e:
            print(f"Skipping {distortion}: {e}")

    print("\n=== Test completed ===")


if __name__ == "__main__":
    main()
# interface.py
import os

# Force CPU and disable GPU/Metal to prevent segfaults
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
# Disable Metal on macOS to prevent segfaults
os.environ["TF_METAL"] = "0"
# Force CPU usage
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

import tensorflow as tf
# Explicitly set CPU only
tf.config.set_visible_devices([], 'GPU')

# Enable unsafe deserialization for Lambda layers
import keras
keras.config.enable_unsafe_deserialization()

import numpy as np

# Use a safe, headless plotting backend on macOS (prevents some segfaults)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

DISTORTION_CLASSES = ["stretch", "compression", "sliding", "nonrigid", "curl"]

# --- Build model and load weights separately (avoids segfault) ---
from model import build_unet

print("Building model architecture...")
model = build_unet(
    input_shape=(256, 256, 1),
    num_cond_classes=len(DISTORTION_CLASSES),
    residual=True
)

print("Loading weights...")
try:
    with tf.device('/CPU:0'):
        model.load_weights("elastic_unet_conditional.h5")
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    try:
        # Try the other file
        model.load_weights("elastic_unet_best.h5")
        print("Loaded weights from best model!")
    except Exception as e2:
        print(f"Failed to load either model: {e2}")
        import sys
        sys.exit(1)

# --- Preprocess input image ---
def load_grayscale(path, size=(256, 256)):
    path = str(path)
    img_bytes = tf.io.read_file(path)
    ext = Path(path).suffix.lower()

    if ext == ".bmp":
        # BMP must be decoded as RGB
        img = tf.io.decode_bmp(img_bytes, channels=3)
    elif ext == ".png":
        img = tf.io.decode_png(img_bytes, channels=1)
    elif ext in (".jpg", ".jpeg"):
        img = tf.io.decode_jpeg(img_bytes, channels=1)
    else:
        img = tf.io.decode_image(img_bytes, channels=1, expand_animations=False)

    # Convert to grayscale if it’s RGB
    if img.shape[-1] == 3:
        img = tf.image.rgb_to_grayscale(img)

    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    if size is not None:
        img = tf.image.resize(img, size, method="bilinear")
    return img  # [H,W,1] float32
# --- Build one-hot condition ---
def get_condition_vector(dist_type: str):
    if dist_type not in DISTORTION_CLASSES:
        raise ValueError(f"Unknown distortion type: {dist_type}")
    onehot = np.zeros((1, len(DISTORTION_CLASSES)), dtype=np.float32)
    onehot[0, DISTORTION_CLASSES.index(dist_type)] = 1.0
    return onehot  # shape (1, C)

# --- Inference ---
def reconstruct(img_path, distortion_type):
    # Load & preprocess
    img = load_grayscale(img_path)     # [H,W,1], float32 [0,1]
    img = tf.expand_dims(img, 0)       # [1,H,W,1]

    # Condition vector
    cond = get_condition_vector(distortion_type)  # [1,C]
    cond = tf.constant(cond)  # Convert to tensor

    # Force CPU inference to prevent segfaults
    with tf.device("/CPU:0"):
        restored = model([img, cond], training=False)[0].numpy()   # [H,W,1] float32

    # Convert to uint8 for saving
    restored_uint8 = tf.image.convert_image_dtype(restored, tf.uint8, saturate=True)
    return restored, restored_uint8

if __name__ == "__main__":
    input_path = "sample_distorted.bmp"   # now correctly decoded
    dist_type = "stretch"

    restored, restored_uint8 = reconstruct(input_path, dist_type)

    # Save result
    tf.io.write_file("restored_cond.png", tf.io.encode_png(restored_uint8))

    # Show result (headless backend saves a window-less figure)
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(load_grayscale(input_path).numpy().squeeze(), cmap="gray")
    plt.title(f"Input ({dist_type})")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(restored.squeeze(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

    # On Agg backend, this writes an image instead of opening a GUI window
    plt.tight_layout()
    plt.savefig("viz_compare.png", dpi=150)
    # If you do want a GUI window and you're not headless, switch to the macOSX backend.
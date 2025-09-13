# interface.py
import os
# Optional: force CPU if Metal causes trouble:  USE_CPU=1 python interface.py
if os.environ.get("USE_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # TF respects this on mac too

# Quiet TF logs a bit
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf
import numpy as np

# Use a safe, headless plotting backend on macOS (prevents some segfaults)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

DISTORTION_CLASSES = ["stretch", "compression", "sliding", "nonrigid", "curl"]

# --- Load model (no compile needed for inference) ---
# If you saved with Keras 3 / TF 2.16+, .keras format is even safer:
# model = tf.keras.models.load_model("elastic_unet_conditional.keras", compile=False)
model = tf.keras.models.load_model(
    "elastic_unet_conditional.h5",
    compile=False,          # <- important; avoids custom_objects & legacy optimizer state
    safe_mode=False,        # okay since we created the model; no pickled lambdas required
)

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

    # If Metal/GPUs misbehave, uncomment:
    # with tf.device("/CPU:0"):
    #     restored = model([img, cond], training=False)[0].numpy()
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

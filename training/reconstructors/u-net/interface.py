import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DISTORTION_CLASSES = ["stretch", "compression", "sliding", "nonrigid", "curl"]

# --- Load model ---
custom_objects = {"ssim_l1_loss": lambda y_true, y_pred: 0.0}  # placeholder if needed
model = tf.keras.models.load_model(
    "elastic_unet_conditional.h5",
    custom_objects={"ssim_l1_loss": lambda y_true, y_pred: 0.0},
    safe_mode=False,  # allow lambda layers
)
# --- Preprocess input image ---
def load_grayscale(path, size=(256, 256)):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, size)
    return img

# --- Build one-hot condition ---
def get_condition_vector(dist_type: str):
    onehot = np.zeros((1, len(DISTORTION_CLASSES)), dtype=np.float32)
    if dist_type in DISTORTION_CLASSES:
        onehot[0, DISTORTION_CLASSES.index(dist_type)] = 1.0
    else:
        raise ValueError(f"Unknown distortion type: {dist_type}")
    return onehot

# --- Inference ---
def reconstruct(img_path, distortion_type):
    # Load & preprocess
    img = load_grayscale(img_path)  # [H,W,1], float32 [0,1]
    img = tf.expand_dims(img, 0)    # add batch dimension

    # Condition vector
    cond = get_condition_vector(distortion_type)

    # Run model
    restored = model.predict([img, cond], verbose=0)[0]  # [H,W,1]

    # Convert to uint8 for saving
    restored_uint8 = tf.image.convert_image_dtype(restored, tf.uint8, saturate=True)
    return restored, restored_uint8

# --- Example usage ---
input_path = "sample_distorted.bmp"
dist_type = "stretch"   # <-- known distortion from classifier or metadata

restored, restored_uint8 = reconstruct(input_path, dist_type)

# Save result
tf.io.write_file("restored_cond.png", tf.io.encode_png(restored_uint8))

# Show result
plt.subplot(1,2,1)
plt.imshow(load_grayscale(input_path), cmap="gray")
plt.title(f"Input ({dist_type})")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(restored[...,0], cmap="gray")
plt.title("Reconstructed")
plt.axis("off")

plt.show()

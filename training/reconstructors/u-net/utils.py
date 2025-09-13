import os
import numpy as np

# Define your distortion classes (must match training order)
DISTORTION_CLASSES = ["affine_warp", "compression_loss", "elastic", "gaussian_noise", "partial_loss",
                      "perspective_warp", "radial", "ridge_erosion", "rotate", "salt_pepper_noise",
                      "scale", "speckle_noise", "stretch", "translate"
                      ]


def build_cond_map(distorted_dir: str):
    """
    Scan a folder of distorted fingerprint images and build a cond_map dictionary:
        filename -> one-hot vector
    Assumes filenames contain the distortion keyword, e.g. 'stretched_101.bmp'.

    Args:
        distorted_dir (str): path to the distorted image folder
    Returns:
        dict: mapping {filename: onehot_vector}
    """
    cond_map = {}
    for fname in os.listdir(distorted_dir):
        f_lower = fname.lower()
        if not f_lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue  # skip non-images

        onehot = np.zeros(len(DISTORTION_CLASSES), dtype=np.float32)
        matched = False
        for idx, cls in enumerate(DISTORTION_CLASSES):
            if cls in f_lower:
                onehot[idx] = 1.0
                matched = True
                break

        if not matched:
            raise ValueError(f"Could not match filename '{fname}' to classes {DISTORTION_CLASSES}")

        cond_map[fname] = onehot

    return cond_map


# --- Example usage ---
if __name__ == "__main__":
    distorted_folder = "data/distorted"
    cond_map = build_cond_map(distorted_folder)

    # Print a few entries
    for k, v in list(cond_map.items())[:5]:
        print(k, "->", v)

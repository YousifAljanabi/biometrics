import os
import numpy as np

DISTORTION_CLASSES = ["affine_warp", "compression_loss", "elastic", "gaussian_noise", "partial_loss",
                      "perspective_warp", "radial", "ridge_erosion", "rotate", "salt_pepper_noise",
                      "scale", "speckle_noise", "stretch", "translate"
                      ]


def build_cond_map(distorted_dir: str):
    """
    Scan folders of distorted fingerprint images and build a cond_map dictionary:
        filename -> one-hot vector
    Assumes directory structure: distorted_dir/{distortion_type}/*.png
    with filenames like: {distortion_name}_{image_name}_var{variation:02d}.png

    Args:
        distorted_dir (str): path to the distorted image folder root
    Returns:
        dict: mapping {filename: onehot_vector}
    """
    cond_map = {}

    # Iterate through distortion type folders
    for distortion_folder in os.listdir(distorted_dir):
        distortion_path = os.path.join(distorted_dir, distortion_folder)
        if not os.path.isdir(distortion_path):
            continue

        # Find matching distortion class
        matched_idx = None
        for idx, cls in enumerate(DISTORTION_CLASSES):
            if cls == distortion_folder:
                matched_idx = idx
                break

        if matched_idx is None:
            print(f"Warning: Unknown distortion folder '{distortion_folder}', skipping")
            continue

        # Create one-hot vector for this distortion
        onehot = np.zeros(len(DISTORTION_CLASSES), dtype=np.float32)
        onehot[matched_idx] = 1.0

        # Add all images in this distortion folder
        for fname in os.listdir(distortion_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                # Store with the relative path from distorted_dir
                relative_path = os.path.join(distortion_folder, fname)
                cond_map[relative_path] = onehot

    return cond_map


# --- Example usage ---
if __name__ == "__main__":
    distorted_folder = "data/distorted"
    cond_map = build_cond_map(distorted_folder)

    # Print a few entries
    for k, v in list(cond_map.items())[:5]:
        print(k, "->", v)

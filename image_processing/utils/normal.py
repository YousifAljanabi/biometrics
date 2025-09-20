# normal.py
import os
import cv2
import numpy as np

# IITK utils
from utils.normalization import normalize as iitk_normalize
from utils.segmentation import create_segmented_and_variance_images
from utils.orientation import calculate_angles
from utils.frequency import ridge_freq
from utils.gabor_filter import gabor_filter

# -------------------------
# Distortion type groups
# -------------------------
DIST_TYPES = {
    "geometric": {"translation","translate","scale","stretch","affine_warp","perspective_warp","radial"},
    "elastic":   {"elastic","elastic_warp"},
    "noise":     {"gaussian_noise","speckle_noise","salt_pepper_noise","compression_loss"},
    "occlusion": {"partial_loss","ridge_erosion"},
}

# -------------------------
# Utilities
# -------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def pad_to_square(img: np.ndarray, size: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

# -------------------------
# Distortion restoration
# -------------------------
def correct_geometric(img): return pad_to_square(to_gray(img), 512)
def correct_elastic_warp(img): return pad_to_square(to_gray(img), 512)
def correct_occlusion(img): return pad_to_square(to_gray(img), 512)

def correct_blur_or_noise(img):
    g = to_gray(img)
    g = cv2.fastNlMeansDenoising(g, None, h=10,
                                 templateWindowSize=7, searchWindowSize=21)
    return pad_to_square(g, 512)

def route_and_restore(img, distortion_type: str):
    dt = (distortion_type or "unknown").lower()
    if dt in DIST_TYPES["geometric"]: return correct_geometric(img)
    if dt in DIST_TYPES["elastic"]:   return correct_elastic_warp(img)
    if dt in DIST_TYPES["noise"]:     return correct_blur_or_noise(img)
    if dt in DIST_TYPES["occlusion"]: return correct_occlusion(img)
    print(f"[WARN] Unknown distortion type '{dt}' → fallback.")
    return correct_geometric(img)

# -------------------------
# IITK-style enhancement
# -------------------------
def iitk_pipeline(img: np.ndarray, block_size: int = 16):
    gray = to_gray(img)

    normalized = iitk_normalize(gray.copy(), 100.0, 100.0)
    segmented, normim, mask = create_segmented_and_variance_images(normalized, block_size, 0.2)
    angles = calculate_angles(normalized, W=block_size, smoth=False)
    freq = ridge_freq(normim, mask, angles, block_size,
                      kernel_size=5, minWaveLength=5, maxWaveLength=15)

    gabor_img = gabor_filter(normim, angles, freq)
    gabor_img = np.nan_to_num(gabor_img)
    gabor_img = cv2.normalize(gabor_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {"skeleton": gabor_img}

# -------------------------
# Full pipeline
# -------------------------
def full_pipeline(img, distortion_type: str):
    restored = route_and_restore(img, distortion_type)
    enhanced = iitk_pipeline(restored)
    return {"restored": restored, **enhanced}

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to fingerprint image")
    ap.add_argument("--type", required=True, help="Distortion type label")
    ap.add_argument("--outdir", default="out_step", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {args.input}")

    outputs = full_pipeline(img, args.type)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    cv2.imwrite(os.path.join(args.outdir, f"{stem}_{args.type}_restored.png"),
                outputs["restored"])
    cv2.imwrite(os.path.join(args.outdir, f"{stem}_{args.type}_skeleton.png"),
                outputs["skeleton"])

    print(f"[OK] Saved results in {os.path.abspath(args.outdir)}")

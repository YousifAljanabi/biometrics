# normal.py
# Distortion restoration + dynamic enhancement + skeletonization (clean version, no ML)

from __future__ import annotations
import cv2
import numpy as np

# ----------------------------
# Distortion type groups
# ----------------------------
DIST_TYPES = {
    "geometric": {"translation", "translate", "scale", "stretch", "affine_warp", "perspective_warp", "radial"},
    "rotate": {"rotate"},
    "elastic": {"elastic", "elastic_warp"},
    "noise": {"gaussian_noise", "speckle_noise", "salt_pepper_noise", "compression_loss"},
    "occlusion": {"partial_loss", "ridge_erosion"}
}

# ----------------------------
# Utilities
# ----------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def pad_to_square(img: np.ndarray, size: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(img, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_AREA)
    H, W = resized.shape[:2]
    canvas = np.full((size, size), 255, dtype=np.uint8)
    y0 = (size - H) // 2
    x0 = (size - W) // 2
    canvas[y0:y0+H, x0:x0+W] = resized
    return canvas

def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def estimate_global_ridge_orientation(img: np.ndarray) -> float:
    f = to_gray(img)
    f = cv2.GaussianBlur(f, (0, 0), 1.2)
    fx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    fy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    ori = 0.5 * np.arctan2(2.0 * fx * fy, (fx**2 - fy**2 + 1e-6))
    deg = np.degrees(ori)
    vals = deg.ravel()
    vals = (vals + 90.0) % 180.0 - 90.0
    return float(np.median(vals))

# ----------------------------
# Distortion-specific handlers
# ----------------------------
def correct_rotation(img: np.ndarray) -> np.ndarray:
    gray = to_gray(img)
    angle = estimate_global_ridge_orientation(gray)
    deskew = rotate(gray, -angle)
    return pad_to_square(deskew, size=512)

def correct_geometric(img: np.ndarray) -> np.ndarray:
    return pad_to_square(to_gray(img), size=512)

def correct_blur_or_noise(img: np.ndarray) -> np.ndarray:
    g = to_gray(img)
    den = cv2.fastNlMeansDenoising(g, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return pad_to_square(den, 512)

def correct_elastic_warp(img: np.ndarray) -> np.ndarray:
    return pad_to_square(to_gray(img), 512)

def correct_occlusion(img: np.ndarray) -> np.ndarray:
    return pad_to_square(to_gray(img), 512)

# ----------------------------
# Enhancement helpers
# ----------------------------
def normalize(img: np.ndarray) -> np.ndarray:
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def enhance_contrast(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def sharpen(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

def morph_cleanup(img: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def thinning(img: np.ndarray) -> np.ndarray:
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    else:
        return img

# ----------------------------
# Router + Dynamic Enhancement
# ----------------------------
def route_and_restore(img: np.ndarray, distortion_type: str) -> np.ndarray:
    dt = (distortion_type or "unknown").lower()
    if dt in DIST_TYPES["rotate"]:
        restored = correct_rotation(img)
    elif dt in DIST_TYPES["geometric"]:
        restored = correct_geometric(img)
    elif dt in DIST_TYPES["elastic"]:
        restored = correct_elastic_warp(img)
    elif dt in DIST_TYPES["noise"]:
        restored = correct_blur_or_noise(img)
    elif dt in DIST_TYPES["occlusion"]:
        restored = correct_occlusion(img)
    else:
        print(f"[WARN] Unknown distortion type '{dt}'. Using fallback.")
        restored = pad_to_square(to_gray(img), 512)
    return restored

def dynamic_enhance(img: np.ndarray, distortion_type: str) -> np.ndarray:
    dt = distortion_type.lower()
    g = to_gray(img)

    if dt in DIST_TYPES["noise"]:
        g = enhance_contrast(g)
        g = sharpen(g)
        g = morph_cleanup(g)

    elif dt in DIST_TYPES["elastic"]:
        g = sharpen(g)

    elif dt in DIST_TYPES["occlusion"]:
        g = enhance_contrast(g)
        g = morph_cleanup(g)

    return g

# ----------------------------
# Main Pipeline
# ----------------------------
def segment(img: np.ndarray) -> np.ndarray:
    g = to_gray(img)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def full_pipeline(img: np.ndarray, distortion_type: str):
    restored = route_and_restore(img, distortion_type)
    enhanced = dynamic_enhance(restored, distortion_type)
    norm = normalize(enhanced)
    ridge = segment(norm)
    skel = thinning(ridge)
    return {
        "restored": restored,
        "ridge": ridge,
        "skeleton": skel
    }

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse, os, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to fingerprint image")
    ap.add_argument("--type", required=True, help="Distortion type label")
    ap.add_argument("--outdir", default="out_step", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    outputs = full_pipeline(img, args.type)

    def _safe_imwrite(path: str, img: np.ndarray) -> None:
        ok = cv2.imwrite(path, img)
        if not ok:
            print(f"[ERROR] Failed to write file: {path}")
            sys.exit(1)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    _safe_imwrite(os.path.join(args.outdir, f"{stem}_{args.type}_restored.png"), outputs["restored"])
    _safe_imwrite(os.path.join(args.outdir, f"{stem}_{args.type}_ridge.png"), outputs["ridge"])
    _safe_imwrite(os.path.join(args.outdir, f"{stem}_{args.type}_skeleton.png"), outputs["skeleton"])

    print(f"Saved outputs in {os.path.abspath(args.outdir)}")

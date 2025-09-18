import cv2
import numpy as np

# ---------- core ops ----------
def load_and_normalize(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {input_path}.")
    return img

def illumination_normalize(img, sigma=25):
    """Remove shading and lighting variations"""
    f = img.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), sigma)
    norm = (f / (bg + 1e-6)) * 128.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def normalize_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE for local contrast improvement"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def denoise_normalized(img, method="median", ksize=3):
    """Reduce noise while preserving edges"""
    if method == "median":
        return cv2.medianBlur(img, ksize)
    elif method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    else:
        return img

# ---------- binarization ----------
def sauvola_or_adaptive(img, win=31, k=0.2, invert=True):
    """Local thresholding"""
    try:
        import cv2.ximgproc as xip
        th = xip.niBlackThreshold(
            img, 255,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
            win, k, xip.SAUVOLA
        )
    except Exception:
        th = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
            max(3, win | 1), 2
        )
    return th

# ---------- resizing ----------
def resize_and_normalize(img, size=256):
    """Resize to fixed square with padding"""
    h, w = img.shape
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.ones((size, size), dtype=np.uint8) * 255
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

# ---------- detection ----------
def detect_source(img, variance_threshold=10):
    """Distinguish scanner vs paper-photo by background uniformity"""
    h, w = img.shape
    patch = img[:min(50, h//4, w//4), :min(50, h//4, w//4)]
    return "paper" if np.std(patch) > variance_threshold else "scanner"

# ---------- processing for paper prints ----------
def process_paper(img):
    """Remove paper grain and extract fingerprint ridges"""
    img = illumination_normalize(img, sigma=25)
    img = normalize_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8))
    img = denoise_normalized(img, method="median", ksize=5)

    bin_img = sauvola_or_adaptive(img, win=51, k=0.3, invert=True)

    # Clean up with morphology
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)  # remove specks
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel) # strengthen ridges

    # Keep only large connected components (discard paper texture)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    mask = np.zeros_like(bin_img)
    min_area = 1000
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

    return mask

# ---------- processing for scanner captures ----------
def process_scanner(img):
    """Enhance clean sensor scans without removing background"""
    img = normalize_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8))
    img = denoise_normalized(img, method="median", ksize=3)
    bin_img = sauvola_or_adaptive(img, win=21, k=0.15, invert=True)
    return bin_img

# ---------- pipeline ----------
def normalizing_pipeline(input_path, output_path, size=256, mode="auto", preserve_resolution=True):
    img = load_and_normalize(input_path)
    source = detect_source(img) if mode == "auto" else mode  # "scanner" or "paper"
    print(f"Detected source: {source}")

    if source == "paper":
        out = process_paper(img)
    else:  # scanner
        out = process_scanner(img)

    if not preserve_resolution:
        out = resize_and_normalize(out, size)

    cv2.imwrite(output_path, out)

# ---------- main ----------
if __name__ == "__main__":
    input_path = "/home/shanshal/Projects/Python/biometrics/image_processing/examples/easy/img.png"
    output_path = "/home/shanshal/Projects/Python/biometrics/image_processing/examples/normalized_fingerprint.jpg"
    normalizing_pipeline(input_path, output_path, size=256, mode="auto", preserve_resolution=True)
    print("Saved:", output_path)


# Option 2 
#
# import cv2
# import numpy as np
#
# def load_and_normalize(input_path):
#     """Loads grayscale fingerprint image"""
#     img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"Could not read image at {input_path}. Check path or file integrity.")
#     return img
#
# def resize_and_normalize(img, size=256):
#     """Resize while keeping aspect ratio, pad to square"""
#     h, w = img.shape
#     scale = size / max(h, w)
#     new_w, new_h = int(w * scale), int(h * scale)
#     resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
#     canvas = np.ones((size, size), dtype=np.uint8) * 255
#     x_off = (size - new_w) // 2
#     y_off = (size - new_h) // 2
#     canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
#     return canvas
#
# def normalize_contrast(img, clip_limit=2.0, tile_grid_size=(8,8)):
#     """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     return clahe.apply(img)
#
# def denoise_normalized(img, method="median", ksize=3):
#     """Denoise image with chosen filter"""
#     if method == "median":
#         return cv2.medianBlur(img, ksize)
#     elif method == "gaussian":
#         return cv2.GaussianBlur(img, (ksize, ksize), 0)
#     else:
#         return img
#
# def remove_background(img):
#     """Remove paper background, keep only fingerprint ridges"""
#     binary = cv2.adaptiveThreshold(
#         img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, 11, 2
#     )
#     kernel = np.ones((3, 3), np.uint8)
#     clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
#     clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
#
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
#     if num_labels > 1:
#         largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#         mask = (labels == largest).astype(np.uint8) * 255
#         fingerprint_only = cv2.bitwise_and(clean, clean, mask=mask)
#         return fingerprint_only
#     else:
#         return clean
#
# def detect_source(img, variance_threshold=10):
#     """Detect source based on background variance"""
#     h, w = img.shape
#     patch_size = min(50, h//4, w//4)  # safe patch size
#     corner_patch = img[:patch_size, :patch_size]
#     background_std = np.std(corner_patch)
#
#     if background_std > variance_threshold:
#         return "phone"
#     else:
#         return "sensor"
#
# def normalizing_pipeline(input_path, output_path, size=256):
#     """Pipeline that adapts depending on image source"""
#     img = load_and_normalize(input_path)
#     source = detect_source(img)
#     print(f"Detected source: {source}")
#
#     img = resize_and_normalize(img, size)
#     img = normalize_contrast(img)
#     img = denoise_normalized(img)
#
#     if source == "phone":
#         img = remove_background(img)
#
#     cv2.imwrite(output_path, img)
#
# if __name__ == "__main__":
#     input_path = "/home/shanshal/Projects/Python/biometrics/image_processing/examples/easy/sample.jpg"
#     output_path = "/home/shanshal/Projects/Python/biometrics/image_processing/examples/normalized_fingerprint.jpg"
#     normalizing_pipeline(input_path, output_path, size=256)
#     print("Saved:", output_path)

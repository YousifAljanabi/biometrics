import os
import cv2
import numpy as np

def load_and_normalize(input_path):
    """Loads grayscale fingerprint image"""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    return img  # or normalize if needed

def resize_image(img, size=256):
    h, w = img.shape
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.ones((size, size), dtype=np.uint8) * 255
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def enhance_contrast(img, clip_limit=2.0, tile_grid_size=(8,8)):
    """Contrast enhancement using CLAHE"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def denoise(img, method="median", ksize=3):
    """Denoise image with chosen filter"""
    if method == "median":
        return cv2.medianBlur(img, ksize)
    elif method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    else:
        return img

def preprocess_pipeline(input_path, output_path, size=256):
    """Full preprocessing pipeline for one image"""
    img = load_and_normalize(input_path)
    img = resize_image(img, size)
    img = enhance_contrast(img)
    img = denoise(img)
    cv2.imwrite(output_path, img)

def preprocess_folder(input_folder, output_folder, size=256):
    """Process all images in a folder and save enhanced copies"""
    os.makedirs(output_folder, exist_ok=True)
    processed_files = []

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        if not os.path.isfile(input_path):
            continue
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue

        output_path = os.path.join(output_folder, filename)
        preprocess_pipeline(input_path, output_path, size)
        processed_files.append(output_path)

    return processed_files

# ---------------------------
# Usage
# ---------------------------
if __name__ == "__main__":
    input_folder = "/home/shanshal/Projects/Python/biometrics/image_processing/examples/easy"
    output_folder = "/home/shanshal/Projects/Python/biometrics/image_processing/examples/enhanced_fingerprints"

    results = preprocess_folder(input_folder, output_folder, size=256)
    print("Processed files:", results)

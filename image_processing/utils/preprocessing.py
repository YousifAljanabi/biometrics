import cv2
import numpy as np

def load_and_normalize(input_path):
    """
    Loads grayscale fingerprint image
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    return img  # change to img.astype(np.float32) / 255.0 if used directly in a model

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
    """
    Contrast enhancement using CLAHE
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def denoise(img, method="median", ksize=3):
    """
    Denoise image with chosen filter
    """
    if method == "median":
        return cv2.medianBlur(img, ksize)
    elif method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    else:
        return img

def preprocess_pipeline(input_path, output_path, size=256:
    img = load_and_normalize(input_path)
    img = resize_image(img, size)
    img = enhance_contrast(img)
    img = denoise(img)
    cv2.imwrite(output_path, img)

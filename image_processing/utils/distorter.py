import cv2
import numpy as np
import random

# -------------------------------
# Geometric Distortions
# -------------------------------

def radial(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    # Random distortion strength
    k1 = random.uniform(-0.00001, 0.00001)  # smaller for subtle effect
    k2 = random.uniform(-0.000001, 0.000001)
    # Camera matrix
    fx = w
    fy = h
    cx = w / 2
    cy = h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, k2, 0, 0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderValue=255)
    cv2.imwrite(output_path, dst)

def stretch(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    scale_x = random.uniform(0.9, 1.1)
    scale_y = random.uniform(0.9, 1.1)
    dst = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, dst)

def rotate(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)
    cv2.imwrite(output_path, rotated)

def translate(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    shift_x = random.randint(-10, 10)
    shift_y = random.randint(-10, 10)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)
    cv2.imwrite(output_path, shifted)

def scale(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    fx = random.uniform(0.9, 1.1)
    fy = random.uniform(0.9, 1.1)
    scaled = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, scaled)

def affine_warp(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    pts1 = np.float32([[w*0.2, h*0.2], [w*0.8, h*0.2], [w*0.2, h*0.8]])
    pts2 = pts1 + np.random.randint(-20, 20, pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    warped = cv2.warpAffine(img, M, (w, h), borderValue=255)
    cv2.imwrite(output_path, warped)

def perspective_warp(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    pts2 = pts1 + np.random.randint(-30, 30, pts1.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (w, h), borderValue=255)
    cv2.imwrite(output_path, warped)

# -------------------------------
# Fingerprint Degradation
# -------------------------------

def ridge_erosion(input_path, output_path, severity=None, iterations=1):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if severity is None:
        severity = random.choice([1, 2])
    kernel_size = 2 * severity + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=iterations)
    cv2.imwrite(output_path, eroded)

def partial_loss(input_path, output_path, patch_size=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    if patch_size is None:
        patch_size = random.randint(20, 40)
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    erased = img.copy()
    erased[y:y+patch_size, x:x+patch_size] = 255  # white patch
    cv2.imwrite(output_path, erased)

def elastic(input_path, output_path, alpha=None, sigma=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if alpha is None:
        alpha = random.uniform(20, 50)
    if sigma is None:
        sigma = random.uniform(4, 6)
    random_state = np.random.RandomState(None)
    dx = cv2.GaussianBlur((random_state.rand(*img.shape) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*img.shape) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    deformed = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    cv2.imwrite(output_path, deformed)

# -------------------------------
# Noise & Compression
# -------------------------------

def add_gaussian_noise(input_path, output_path, mean=0, std=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if std is None:
        std = random.uniform(5, 15)
    noise = np.random.normal(mean, std, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy)

def add_salt_pepper_noise(input_path, output_path, amount=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if amount is None:
        amount = random.uniform(0.001, 0.01)
    noisy = img.copy()
    num_pixels = int(amount * img.size)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in img.shape]
    for i in range(num_pixels):
        if np.random.rand() < 0.5:
            noisy[coords[0][i], coords[1][i]] = 0
        else:
            noisy[coords[0][i], coords[1][i]] = 255
    cv2.imwrite(output_path, noisy)

def add_speckle_noise(input_path, output_path, multiplier=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if multiplier is None:
        multiplier = random.uniform(0.05, 0.15)
    noise = np.random.randn(*img.shape) * multiplier
    noisy = np.clip(img + img * noise, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy)

def compression_loss(input_path, output_path, quality=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if quality is None:
        quality = random.randint(40, 80)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(output_path, decoded)

import cv2
import numpy as np
import random

def radial(input_path, output_path, k1=None, k2=None):
    """Apply radial distortion to simulate lens effects."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    # Realistic random values for lens distortion
    if k1 is None:
        k1 = random.uniform(-0.00001, 0.00001)
    if k2 is None:
        k2 = random.uniform(-0.000001, 0.000001)

    fx = w
    fy = h
    cx = w / 2
    cy = h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, k2, 0, 0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderValue=255)
    cv2.imwrite(output_path, dst)

def stretch(input_path, output_path, scale_x=None, scale_y=None):
    """Apply non-uniform scaling (stretching) to fingerprint."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Realistic stretch values - modest distortion
    if scale_x is None:
        scale_x = random.uniform(0.85, 1.15)
    if scale_y is None:
        scale_y = random.uniform(0.85, 1.15)

    dst = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, dst)

def rotate(input_path, output_path, angle=None):
    """Apply rotation to fingerprint."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    # Realistic rotation angles - moderate rotation
    if angle is None:
        angle = random.uniform(-15, 15)

    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)
    cv2.imwrite(output_path, rotated)

def translate(input_path, output_path, shift_x=None, shift_y=None):
    """Apply translation (shifting) to fingerprint."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Realistic translation values - modest shifting
    if shift_x is None:
        shift_x = random.randint(-20, 20)
    if shift_y is None:
        shift_y = random.randint(-20, 20)

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)
    cv2.imwrite(output_path, shifted)

def scale(input_path, output_path, fx=None, fy=None):
    """Apply uniform scaling to fingerprint."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Realistic scaling factors - modest uniform scaling
    if fx is None:
        fx = random.uniform(0.8, 1.2)
    if fy is None:
        fy = fx  # Keep uniform scaling by default

    scaled = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, scaled)

def affine_warp(input_path, output_path, warp_strength=None):
    """Apply affine transformation to simulate fingerprint deformation."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    # Realistic warp strength - moderate deformation
    if warp_strength is None:
        warp_strength = random.randint(10, 30)

    pts1 = np.float32([[w*0.2, h*0.2], [w*0.8, h*0.2], [w*0.2, h*0.8]])
    pts2 = pts1 + np.random.randint(-warp_strength, warp_strength, pts1.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(pts1, pts2)
    warped = cv2.warpAffine(img, matrix, (w, h), borderValue=255)
    cv2.imwrite(output_path, warped)

def perspective_warp(input_path, output_path, warp_strength=None):
    """Apply perspective transformation to simulate viewing angle changes."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    # Realistic perspective warp - moderate viewing angle changes
    if warp_strength is None:
        warp_strength = random.randint(15, 40)

    pts1 = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    pts2 = pts1 + np.random.randint(-warp_strength, warp_strength, pts1.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderValue=255)
    cv2.imwrite(output_path, warped)

# -------------------------------
# Fingerprint Degradation
# -------------------------------

def ridge_erosion(input_path, output_path, severity=None, iterations=None):
    """Apply ridge erosion to simulate fingerprint wear or poor capture quality."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Realistic erosion parameters - subtle ridge thinning
    if severity is None:
        severity = random.choice([1, 2, 3])
    if iterations is None:
        iterations = random.choice([1, 2])

    kernel_size = 2 * severity + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=iterations)
    cv2.imwrite(output_path, eroded)

def partial_loss(input_path, output_path, patch_size=None, num_patches=None):
    """Apply partial loss to simulate damaged or missing fingerprint regions."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]

    # Realistic patch parameters - small to medium missing areas
    if patch_size is None:
        patch_size = random.randint(15, 50)
    if num_patches is None:
        num_patches = random.randint(1, 3)

    erased = img.copy()
    for _ in range(num_patches):
        x = random.randint(0, max(0, w - patch_size))
        y = random.randint(0, max(0, h - patch_size))
        erased[y:y+patch_size, x:x+patch_size] = 255  # white patch
    cv2.imwrite(output_path, erased)

def elastic(input_path, output_path, alpha=None, sigma=None):
    """Apply elastic deformation to simulate skin deformation during capture."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Realistic elastic deformation parameters - moderate skin distortion
    if alpha is None:
        alpha = random.uniform(15, 40)
    if sigma is None:
        sigma = random.uniform(3, 7)

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

def add_gaussian_noise(input_path, output_path, mean=None, std=None):
    """Add Gaussian noise to simulate sensor noise or poor lighting."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Realistic noise parameters - subtle to moderate noise
    if mean is None:
        mean = 0
    if std is None:
        std = random.uniform(3, 12)

    noise = np.random.normal(mean, std, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy)

def add_salt_pepper_noise(input_path, output_path, amount=None):
    """Add salt and pepper noise to simulate digital artifacts or dust."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Realistic salt & pepper noise - very sparse noise
    if amount is None:
        amount = random.uniform(0.0005, 0.005)

    noisy = img.copy()
    num_pixels = int(amount * img.size)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in img.shape]
    for i in range(num_pixels):
        if np.random.rand() < 0.5:
            noisy[coords[0][i], coords[1][i]] = 0  # pepper (black)
        else:
            noisy[coords[0][i], coords[1][i]] = 255  # salt (white)
    cv2.imwrite(output_path, noisy)

def add_speckle_noise(input_path, output_path, multiplier=None):
    """Add speckle noise to simulate multiplicative noise from imaging systems."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Realistic speckle noise - moderate multiplicative noise
    if multiplier is None:
        multiplier = random.uniform(0.03, 0.12)

    noise = np.random.randn(*img.shape) * multiplier
    noisy = np.clip(img + img * noise, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy)

def compression_loss(input_path, output_path, quality=None):
    """Apply JPEG compression to simulate compression artifacts."""
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Realistic compression quality - moderate to low quality
    if quality is None:
        quality = random.randint(30, 75)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(output_path, decoded)

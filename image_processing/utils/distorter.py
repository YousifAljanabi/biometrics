import os
import random
import cv2
import numpy as np
from wand.image import Image

def radial(input_path, output_path):
    distortion_amounts = (
        random.uniform(0.95, 1.05),  # barrel/pincushion
        random.uniform(-0.1, 0.1),   # horizontal skew
        random.uniform(-0.1, 0.1),   # vertical skew
        1.0
    )
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.distort('barrel', distortion_amounts)
        img.save(filename=output_path)

def stretch(input_path, output_path):
    scale_x = random.uniform(0.9, 1.1)
    scale_y = random.uniform(0.9, 1.1)
    with Image(filename=input_path) as img:
        img.resize(int(img.width * scale_x), int(img.height * scale_y))
        img.save(filename=output_path)

def rotate(input_path, output_path):
    angle = random.uniform(-10, 10)
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.rotate(angle)
        img.save(filename=output_path)

def translate(input_path, output_path, shift_x=None, shift_y=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if shift_x is None:
        shift_x = random.randint(-10, 10)
    if shift_y is None:
        shift_y = random.randint(-10, 10)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    cv2.imwrite(output_path, shifted)

def scale(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    new_width = int(image.shape[1] * random.uniform(0.9, 1.1))
    new_height = int(image.shape[0] * random.uniform(0.9, 1.1))
    resized_image = cv2.resize(image, (new_width, new_height))
    cv2.imwrite(output_path, resized_image)

def affine_warp(input_path, output_path):
    w = random.randint(30, 50)
    h = random.randint(30, 50)
    src_points = [[w, w], [w, h], [h, w]]
    dst_points = [[sx + random.randint(-10, 10), sy + random.randint(-10, 10)]
                  for sx, sy in src_points]
    args = []
    for (sx, sy), (dx, dy) in zip(src_points, dst_points):
        args.extend([sx, sy, dx, dy])
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.distort('affine', tuple(args))
        img.save(filename=output_path)

def perspective_warp(input_path, output_path):
    w = random.randint(30, 50)
    h = random.randint(30, 50)
    src_points = [[0, 0], [w, 0], [w, h], [0, h]]
    dst_points = [[sx + random.randint(-10, 10), sy + random.randint(-10, 10)]
                  for sx, sy in src_points]
    args = []
    for (sx, sy), (dx, dy) in zip(src_points, dst_points):
        args.extend([sx, sy, dx, dy])
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.distort('perspective', tuple(args))
        img.save(filename=output_path)

def ridge_erosion(input_path, output_path, severity=None, iterations=1):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if severity is None:
        severity = random.choice([1, 2])  # kernel
    kernel_size = 2 * severity + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosed_img = cv2.erode(img, kernel, iterations=iterations)
    cv2.imwrite(output_path, erosed_img)

def partial_loss(input_path, output_path, patch_size=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    if patch_size is None:
        patch_size = random.randint(20, 40)
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    mask = np.ones_like(img) * 255
    cv2.rectangle(mask, (x, y), (x+patch_size-1, y+patch_size-1), 0, -1)
    erased_img = cv2.bitwise_and(img, mask)
    cv2.imwrite(output_path, erased_img)

def elastic(input_path, output_path, alpha=None, sigma=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if alpha is None:
        alpha = random.uniform(20, 50)
    if sigma is None:
        sigma = random.uniform(4, 6)
    random_state = np.random.RandomState(None)
    dx = random_state.rand(*img.shape) * 2 - 1
    dy = random_state.rand(*img.shape) * 2 - 1
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    deformed = cv2.remap(img, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    cv2.imwrite(output_path, deformed)

def add_gaussian_noise(input_path, output_path, mean=0, std=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if std is None:
        std = random.uniform(5, 15)
    noise = np.random.normal(mean, std, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
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
    noisy = img + img * noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy)

def compression_loss(input_path, output_path, quality=None):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if quality is None:
        quality = random.randint(40, 80)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(output_path, decoded)

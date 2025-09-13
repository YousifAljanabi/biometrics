import os
import random
from wand.image import Image
import cv2
import numpy as np

def radial(input_path, output_path):
    distortion_amounts = (random.uniform(0.8, 1.2),
                          random.uniform(0.0, 0.5),
                          random.uniform(0.0, 0.5),
                          1.0)
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.distort('barrel', distortion_amounts)
        img.save(filename=output_path)

def stretch(input_path, output_path):
    scale_x = random.uniform(0.8, 1.2)
    scale_y = random.uniform(0.8, 1.2)
    with Image(filename=input_path) as img:
        img.resize(int(img.width * scale_x), int(img.height * scale_y))
        img.save(filename=output_path)

def rotate(input_path, output_path):
    angle = random.uniform(-15, 15)
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.rotate(angle)
        img.save(filename=output_path)

def translate(input_path, output_path, shift_x=None, shift_y=None):
    with Image(filename=input_path) as img:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if shift_x is None:
        shift_x = random.randint(-20, 20)
    if shift_y is None:
        shift_y = random.randint(-20, 20)
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
    w = random.randint(20, 50)
    h = random.randint(20, 50)
    src_points = [[w, w], [w, h], [h, w]]
    dst_points = [[random.randint(0, w*2), random.randint(0, h*2)] for _ in range(3)]
    args = []
    for (sx, sy), (dx, dy) in zip(src_points, dst_points):
        args.extend([sx, sy, dx, dy])
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.distort('affine', tuple(args))
        img.save(filename=output_path)

def perspective_warp(input_path, output_path):
    w = random.randint(20, 50)
    h = random.randint(20, 50)
    src_points = [[0, 0], [w, 0], [w, h], [0, h]]
    dst_points = [[random.randint(0, w*2), random.randint(0, h*2)] for _ in range(4)]
    args = []
    for (sx, sy), (dx, dy) in zip(src_points, dst_points):
        args.extend([sx, sy, dx, dy])
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.distort('perspective', tuple(args))
        img.save(filename=output_path)


def ridge_erosion(input_path, output_path, severity=1, iterations=1):
    """
    1 is the lowest severity
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    kernel_size = 2 * severity + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    erosed_img = cv2.erode(img, kernel, iterations=iterations)
    cv2.imwrite(output_path, erosed_img)


def partial_loss(input_path, output_path, patch_size=50):
    """
    Yeets a random rect from the image
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    mask = np.ones_like(img) * 255
    cv2.rectangle(mask, (x, y), (x+patch_size-1, y+patch_size-1), 0, -1)
    erased_img = cv2.bitwise_and(img, mask)
    cv2.imwrite(output_path, erased_img)

def elastic(input_path, output_path, alpha = 0, sigma  = 0):
    """
    alpha: how far pixels can move
    sigma: smoothness or curve of the actual deformation itself
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    random_state = np.random.RandomState(None)
    dx = random_state.rand(*img.shape) * 2 - 1
    dy = random_state.rand(*img.shape) * 2 - 1

    # Just a filter don't freak out from gauss jumpscare
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    #setting up the grid
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    deformed = cv2.remap(img, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)
    cv2.imwrite(output_path, deformed)

def add_gaussian_noise(input_path, output_path, mean=0, std=10):
    """
    mean: average of noise distrubtion
    std: how strong the noise is default is 10
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    noise = np.random.normal(mean, std, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy)

def add_salt_pepper_noise(input_path, output_path, amount=0.01):
    """
    Straight up adding random 0 or 1 pixels
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    noisy = img.copy()
    num_pixels = int(amount * img.size)
    coords = [np.random.randint(0, i - 1, num_pixels) for i in img.shape]
    for i in range(num_pixels):
        if np.random.rand() < 0.5:
            noisy[coords[0][i], coords[1][i]] = 0
        else:
            noisy[coords[0][i], coords[1][i]] = 255
    cv2.imwrite(output_path, noisy)
def add_speckle_noise(input_path, output_path):
    """
    It's like a multiplicative gaussian?
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    noise = np.random.randn(*img.shape)
    noisy = img + img * noise * 0.1
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, noisy)

import cv2
import numpy as np

def compression_loss(input_path, output_path, quality=30):
    """
    Simulates compression artifacts by saving/reloading as JPEG.
    quality: JPEG quality (1=worst, 100=best). Lower = more artifacts.
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(output_path, decoded)





def apply_random_distortions(input_folder, output_folder, samples_per_image):
    os.makedirs(output_folder, exist_ok=True)
    distortions = [radial, stretch, rotate, translate, scale, affine_warp, perspective_warp, ridge_erosion, partial_loss]
    for root, dirs, _ in os.walk(input_folder):
        if root == input_folder:
            for subdir in dirs:
                sub_input = os.path.join(root, subdir)
                sub_output = os.path.join(output_folder, subdir)
                os.makedirs(sub_output, exist_ok=True)
                for file in os.listdir(sub_input):
                    if file.lower().endswith(('.bmp', '.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                        input_path = os.path.join(sub_input, file)
                        for i in range(samples_per_image):
                            output_path = os.path.join(sub_output, f"{os.path.splitext(file)[0]}_dist{i}.bmp")
                            chosen = random.sample(distortions, k=random.randint(1, 2))
                            temp_path = input_path
                            for j, distortion in enumerate(chosen):
                                final_path = output_path if j == len(chosen) - 1 else output_path + f"_tmp{j}.bmp"
                                distortion(temp_path, final_path)
                                temp_path = final_path

input_folder = '/home/shanshal/Projects/Python/biometrics/image_processing/examples'
output_folder = '/home/shanshal/Projects/Python/biometrics/image_processing/output/distorted_samples'

apply_random_distortions(input_folder, output_folder, samples_per_image=1)

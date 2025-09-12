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
    scale_x = random.uniform(0.8, 1.5)
    scale_y = random.uniform(0.8, 1.5)
    with Image(filename=input_path) as img:
        img.resize(int(img.width * scale_x), int(img.height * scale_y))
        img.save(filename=output_path)

def rotate(input_path, output_path):
    angle = random.uniform(-45, 45)
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.rotate(angle)
        img.save(filename=output_path)

def translate(input_path, output_path, shift_x=0, shift_y=0):
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        # affine transform for translation: x' = x + shift_x, y' = y + shift_y
        args = [0, 0, shift_x, shift_y,
                img.width, 0, img.width + shift_x, shift_y,
                0, img.height, shift_x, img.height + shift_y]
        img.distort('affine', tuple(args))
        img.save(filename=output_path)


def scale(input_path, output_path):
    image = cv2.imread(input_path)
    new_width = int(image.shape[1] * random.uniform(0.8, 1.5))
    new_height = int(image.shape[0] * random.uniform(0.8, 1.5))
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

def apply_random_distortions(input_folder, output_folder, samples_per_image):
    os.makedirs(output_folder, exist_ok=True)
    distortions = [radial, stretch, rotate, translate, scale, affine_warp, perspective_warp]

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
                                final_path = output_path if j == len(chosen) - 1 else temp_path + f"_tmp{j}.bmp"
                                distortion(temp_path, final_path)
                                temp_path = final_path

input_folder = '/home/shanshal/Projects/Python/biometrics/image_processing/examples'
output_folder = '/home/shanshal/Projects/Python/biometrics/image_processing/output/distorted_samples'

apply_random_distortions(input_folder, output_folder, samples_per_image=1)

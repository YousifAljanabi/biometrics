import os
from distorter import (
    radial, stretch, rotate, translate, scale,
    affine_warp, perspective_warp,
    ridge_erosion, partial_loss, elastic,
    add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise,
    compression_loss
)
from preprocessing import preprocess_pipeline


def generate_testing_samples(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    distortions = [
        radial, stretch, rotate, translate, scale,
        affine_warp, perspective_warp,
        ridge_erosion, partial_loss, elastic,
        add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise,
        compression_loss
    ]
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.bmp', '.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, file)
            base_name = os.path.splitext(file)[0]
            for distortion in distortions:
                output_path = os.path.join(output_folder, f"{base_name}_{distortion.__name__}.bmp")
                distortion(input_path, output_path)

def preprocess_testing_samples(input_folder, output_folder, size=(256,256)):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.bmp', '.tif', '.tiff', '.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            preprocess_pipeline(input_path, output_path, size)

examples_folder = '/home/shanshal/Projects/Python/biometrics/image_processing/examples'
testing_samples = '/home/shanshal/Projects/Python/biometrics/image_processing/output/testing_samples'
testing_samples_normal = '/home/shanshal/Projects/Python/biometrics/image_processing/output/testing_samples_normal'

generate_testing_samples(examples_folder, testing_samples)
preprocess_testing_samples(testing_samples, testing_samples_normal)

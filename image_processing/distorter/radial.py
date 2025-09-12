import os
from wand.image import Image

input_folder = "/home/shanshal/Projects/Python/biometrics/image_processing/test_in/DB2_B bmp"
output_folder = "/home/shanshal/Projects/Python/biometrics/image_processing/test_in_barrel"
distortion_amounts = (0.9, 0.4, 0.2, 1.0)  # barrel distortion parameters (a, b, c, d)

for root, dirs, files in os.walk(input_folder):
    relative_path = os.path.relpath(root, input_folder)
    output_root = os.path.join(output_folder, relative_path)
    os.makedirs(output_root, exist_ok=True)

    for filename in files:
        if filename.lower().endswith(".bmp"):
            input_path = os.path.join(root, filename)
            output_path = os.path.join(output_root, filename)
            with Image(filename=input_path) as img:
                img.virtual_pixel = 'white'  # fill empty areas with white
                img.distort('barrel', distortion_amounts)
                img.save(filename=output_path)

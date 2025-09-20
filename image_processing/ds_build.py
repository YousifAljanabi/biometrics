# import os
# import cv2
# from distorters import (
#     radial, stretch, rotate, translate, scale,
#     affine_warp, perspective_warp,
#     ridge_erosion, partial_loss, elastic,
#     add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise,
#     compression_loss
# )
# import time
#
# # Root paths
# input_folder = "/home/shanshal/Projects/Python/biometrics/image_processing/examples/clean"
# output_root = "/home/shanshal/Projects/Python/biometrics/image_processing/distorted_db_labeled"
# distortions = {
#     "radial": radial,
#     "stretch": stretch,
#     "rotate": rotate,
#     "translate": translate,
#     "scale": scale,
#     "affine_warp": affine_warp,
#     "perspective_warp": perspective_warp,
#     "ridge_erosion": ridge_erosion,
#     "partial_loss": partial_loss,
#     "elastic": elastic,
#     "gaussian_noise": add_gaussian_noise,
#     "salt_pepper_noise": add_salt_pepper_noise,
#     "speckle_noise": add_speckle_noise,
#     "compression_loss": compression_loss,
# }
#
# # Ensure output directories
# for name in distortions.keys():
#     os.makedirs(os.path.join(output_root, name), exist_ok=True)
#
# start = time.time()
#
# # Collect and slice .bmp files
# files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".bmp")])
# half = len(files) // 1
# files = files[:half]   # take only half
# for file in files:
#     input_path = os.path.join(input_folder, file)
#     base_name = os.path.splitext(file)[0]
#     for name, fn in distortions.items():
#         output_path = os.path.join(output_root, name, f"{name}_{base_name}.bmp")
#         fn(input_path, output_path)
#
# end = time.time()
# print(f"Process finished in {end - start:.2f} seconds")

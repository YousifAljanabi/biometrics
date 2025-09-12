import os
from PIL import Image

def convert_tif_to_bmp(input_folder):
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")

    # Create output folder
    folder_name = os.path.basename(os.path.normpath(input_folder))
    output_folder = os.path.join(os.path.dirname(input_folder), f"{folder_name} bmp")
    os.makedirs(output_folder, exist_ok=True)

    # Process each .tif file
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".tif", ".tiff")):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".bmp"
            output_path = os.path.join(output_folder, output_filename)
            with Image.open(input_path) as img:
                img.save(output_path, "BMP")
            print(f"Converted: {input_path} -> {output_path}")

    print(f"All .tif files converted. Saved in: {output_folder}")

convert_tif_to_bmp("/home/shanshal/Projects/Python/biometrics/assets/DB2_B")

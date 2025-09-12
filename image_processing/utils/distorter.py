import os
import numpy as np
from wand.image import  Image


input_folder = ""
output_folder = ""


def radial_distort(distortion_amounts, input_path, output_path):
    """
    distortion_amounts = (0.9, 0.4, 0.2, 1.0)
    """
    with Image(filename = input_path) as img:
        img.virtual_pixel = 'white'
        img.distort('barrel', distortion_amounts)
        img.save(filename=output_path)
# NOT IMPLEMENTED YET
def elastic_distort():
    """
    Random elastic deformation
    """
#

def stretch_image_wand(input_path, output_path, scale_x=1.0, scale_y=1.0):
    """
    scales are floats :>
    """
    with Image(filename=input_path) as img:
        new_width = int(img.width * scale_x)
        new_height = int(img.height * scale_y)
        img.resize(width=new_width, height=new_height)
        img.save(filename=output_path)
def affine_warp(input_path, output_path, src_points, dst_points):
    """
    src_points & dst_points: List of 3 [x,y] points in the original AND the output image
    """
    args = []
    for (sx, sy), (dx,dy) in zip(src_points, dst_points):
        args.extend([sx, sy, dx, dy])
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.distort('affine' tuple(args))
        img.save(filename=output_path )
def perspective_warp_wand(input_path, output_path, src_points, dst_points):
    args = []
    for (sx, sy), (dx, dy) in zip(src_points, dst_points):
        args.extend([sx, sy, dx, dy])
    with Image(filename=input_path) as img:
        img.virtual_pixel = 'white'
        img.distort('perspective', tuple(args))
        img.save(filename=output_path)
def rotate.py(input_path, output_path, deg):
    with Image(filename=input_path) as  img:
        img.virtual_pixel = 'white'
        img.rotate(angle)
        img.save(filename=output_path)
def translate_image_wand_simple(input_path, output_path, shift_x=0, shift_y=0):
    with Image(filename=input_path) as img:
        img.transform(resize=None, crop=None, translate=(shift_x, shift_y))
        img.background_color = 'white'
        img.save(filename=output_path)
def scale(input_path, output_path, new_width, new_height):
    image = cv2.imread(input_file)
    resized_image = cv2.resize(image,(new_width, new_height))
    cv2.imwrite(output_file)
    

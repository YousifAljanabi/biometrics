import cv2
import numpy as np

def stretch_image(img, scale_x=1.0, scale_y=1.0):
    """
    Stretch an image along x and y axes.
    
    Parameters:
    - img: np.ndarray, input image
    - scale_x: float, stretching factor along width
    - scale_y: float, stretching factor along height
    
    Returns:
    - stretched_img: np.ndarray, stretched image
    """
    h, w = img.shape[:2]
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)
    
    # Create the stretching matrix
    M = np.array([[scale_x, 0, 0],
                  [0, scale_y, 0]], dtype=np.float32)
    
    stretched_img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    return stretched_img

# Example usage:
image = cv2.imread('../examples/101_1.bmp')
stretched = stretch_image(image, scale_x=1.2, scale_y=0.8)
cv2.imwrite('../output/stretch.bmp', stretched)

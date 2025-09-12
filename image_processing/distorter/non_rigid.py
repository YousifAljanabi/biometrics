import cv2
import numpy as np
def elastic_deform(img, alpha, sigma):
    """
    Apply random elastic deformation on the image
    img: path to the image
    alpha: float that is scaring factor
    sigma: float that is how smooth the deform is????
    """
random_state = np.random.RandomState(None)
shape = img.shape[:2]

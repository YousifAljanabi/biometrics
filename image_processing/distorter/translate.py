import cv2
import numpy as np

image = cv2.imread('../examples/101_1.bmp')
# Displacement
tx, ty = 100, 50  # Shift amount
height, width = image.shape[:2]

translation_matrix = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

cv2.imshow("Translated image", translated_image)
cv2.imwrite("../output/traslate.bmp", translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

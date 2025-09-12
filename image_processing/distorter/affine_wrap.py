import cv2
import numpy as np

# Load input image
img = cv2.imread('../examples/101_1.bmp', cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError("Image not found. Check the path: ./examples/101_1.bmp")

# Get image dimensions
rows, cols, ch = img.shape

# --- Affine Warp ---
# Define 3 points in original image
pts1 = np.float32([[150, 50], [20, 50], [50, 200]])
# Define where those points should map to
pts2 = np.float32([[120, 100], [50, 100], [100, 250]])

# Compute affine transform matrix
M_affine = cv2.getAffineTransform(pts1, pts2)

# Apply warp
warp_affine = cv2.warpAffine(img, M_affine, (cols, rows))

# Display results
cv2.imshow('Original', img)
cv2.imshow('Affine Warp', warp_affine)
cv2.imwrite('../output/warp_affine2.bmp', warp_affine)
cv2.waitKey(0)
cv2.destroyAllWindows()

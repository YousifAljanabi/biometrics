import cv2
import numpy as np

# Load warped affine image
warp_affine = cv2.imread('../output/warp_affine2.bmp')

rows, cols, ch = warp_affine.shape

# Original correspondences
pts1 = np.float32([[150, 50], [20, 50], [50, 200]])
# Define where those points should map to
pts2 = np.float32([[120, 100], [50, 100], [100, 250]])

# Forward affine transform
M_affine = cv2.getAffineTransform(pts1, pts2)
# Inverse affine transform
M_affine_inv = cv2.invertAffineTransform(M_affine)

# Reverse warp
restored_affine = cv2.warpAffine(warp_affine, M_affine_inv, (cols, rows))

# Save result
cv2.imwrite('restored_affine.bmp', restored_affine)

# Display
cv2.imshow('Warped Affine', warp_affine)
cv2.imshow('Restored Affine', restored_affine)
cv2.waitKey(0)
cv2.destroyAllWindows()

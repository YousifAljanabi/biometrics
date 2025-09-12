import cv2
import numpy as np

# Load the previously warped image
img_warped = cv2.imread('../output/persp_warp.bmp')
hh, ww = img_warped.shape[:2]

# Original input quadrilateral and output rectangle from forward warp
input_pts = np.float32([[136,113], [206,130], [173,207], [132,196]])
width = round(np.hypot(input_pts[0,0]-input_pts[1,0], input_pts[0,1]-input_pts[1,1]))
height = round(np.hypot(input_pts[0,0]-input_pts[3,0], input_pts[0,1]-input_pts[3,1]))
x, y = input_pts[0,0], input_pts[0,1]
output_pts = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])

# Compute the reverse perspective matrix
M_rev = cv2.getPerspectiveTransform(output_pts, input_pts)

# Apply reverse warp
restored = cv2.warpPerspective(img_warped, M_rev, (ww, hh),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))

# Save and show the restored image
cv2.imwrite("../output/persp_restored.bmp", restored)
cv2.imshow("Restored Image", restored)
cv2.waitKey(0)
cv2.destroyAllWindows()

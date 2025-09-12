import numpy as np
import cv2
import math

# read input
img = cv2.imread('../examples/101_1.bmp', cv2.IMREAD_COLOR)
hh, ww = img.shape[:2]


# specify input coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x,
input = np.float32([[136,113], [206,130], [173,207], [132,196]])

# get top and left dimensions and set to output dimensions of red rectangle
width = round(math.hypot(input[0,0]-input[1,0], input[0,1]-input[1,1]))
height = round(math.hypot(input[0,0]-input[3,0], input[0,1]-input[3,1]))

# set upper left coordinates for output rectangle
x = input[0,0]
y = input[0,1]

# specify output coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x,
output = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])

# compute perspective matrix
matrix = cv2.getPerspectiveTransform(input,output)
print(matrix)

# Note that output size is the same as the input image size
imgOutput = cv2.warpPerspective(img, matrix, (ww,hh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

# save the warped output
cv2.imwrite("../output/persp_warp.bmp", imgOutput)

# show the result
cv2.imshow("result", imgOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()

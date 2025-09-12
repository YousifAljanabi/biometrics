import cv2

image = cv2.imread("../examples/101_1.bmp")

# Rotate 90 degrees clockwise
rotated_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Rotate 90 degrees counter-clockwise
rotated_90_counterclockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Rotate 180 degrees
rotated_180 = cv2.rotate(image, cv2.ROTATE_180)

# Display the rotated images (optional)
cv2.imshow("Original Image", image)
cv2.imshow("Rotated 90 Clockwise", rotated_90_clockwise)
cv2.imshow("Rotated 90 Counter-Clockwise", rotated_90_counterclockwise)
cv2.imshow("Rotated 180", rotated_180)
cv2.imwrite("../output/rotated90.bmp", rotated_90_clockwise)
cv2.waitKey(0)
cv2.destroyAllWindows()

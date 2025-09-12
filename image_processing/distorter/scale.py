import cv2
 
# Load the image
image = cv2.imread("../examples/101_1.bmp")
 
new_width = 150
new_height = 450
 

resized_image = cv2.resize(image, (new_width, new_height))
 
# Display the resized image
cv2.imshow("Og image", image)
cv2.imshow("Resized Image", resized_image)
cv2.imwrite("../output/scale.bmp", "img")
cv2.waitKey(0)
cv2.destroyAllWindows()

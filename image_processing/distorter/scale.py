import cv2
 
# Load the image
image = cv2.imread("../examples/101_1.bmp")
 
#######################################
# # Display the original image        #
# cv2.imshow("Original Image", image) #
# cv2.waitKey(0)                      #
# cv2.destroyAllWindows()             #
#######################################
# Define different scaling factors for width and height
#fx = 0.088  
#fy = 0.066
new_width = 150
new_height = 450
 

 # We might need this later i have no idea what this is but i found it online
#############################################################
#
## Apply different interpolation methods
# resized_area = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
# resized_linear = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
# resized_cubic = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
# resized_nearest = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
# Display the resized images                              #
# cv2.imshow("Original Image", image)                       #
# cv2.imshow("Resized with INTER_AREA", resized_area)       #
# cv2.imshow("Resized with INTER_LINEAR", resized_linear)   #
# cv2.imshow("Resized with INTER_CUBIC", resized_cubic)     #
# cv2.imshow("Resized with INTER_NEAREST", resized_nearest) #
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Resizing #
#
#############################################################

resized_image = cv2.resize(image, (new_width, new_height))
 
# Display the resized image
cv2.imshow("Og image", image)
cv2.imshow("Resized Image", resized_image)
cv2.imwrite("../output/scale.bmp", "img")
cv2.waitKey(0)
cv2.destroyAllWindows()

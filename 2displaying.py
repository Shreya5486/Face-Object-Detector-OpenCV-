import cv2

image = cv2.imread("python_image.jpg")
if image is not None:
    cv2.imshow("Image showing", image) #open the window
    cv2.waitKey(0) #wait until any key is pressed
    cv2.destroyAllWindows() #close the window
else:
    print("Could not load image.")



import cv2

image = cv2.imread("Image resizing & Shaping/python_image.jpg")

if image is not None:
    cropped = image[100:200, 50:150]
    cv2.imshow("Original", image)
    cv2.imshow("Cropped", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
import cv2

image = cv2.imread("python_image.jpg")
if image is not None:
    success = cv2.imwrite("output_image.jpg", image)
    if success:
        print("Image saved successfully.")
    else:
        print("Could not save the image.")
else:
    print("Could not load image.")
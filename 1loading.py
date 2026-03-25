import cv2

image = cv2.imread('python_image.jpg')

if image is None:
    print("Error: Image not found.")
else:
    print("Image loaded successfully.")
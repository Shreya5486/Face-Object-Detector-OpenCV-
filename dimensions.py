import cv2

image = cv2.imread("python_image.jpg")
if image is not None:
    h, w, c = image.shape
    print(f"Image loaded:\nHeight: {h}\nWidth: {w}\nchannels: {c}")
else:
    print("Could not load image.")
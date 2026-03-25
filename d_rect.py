import cv2

image = cv2.imread("image drawing functions/python_image.jpg")

if image is None:
    print("Oops! Your image is not working")

else:
    print("Image loaded successfully!")

    pt1 = (50, 50)
    pt2 = (250, 200)
    color = (0, 0, 255)
    thickness = 3

    cv2.rectangle(image, pt1, pt2, color, thickness)

    cv2.imshow("Image focusing rectangle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
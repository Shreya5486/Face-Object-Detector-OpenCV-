import cv2

image = cv2.imread("image drawing functions/python_image.jpg")

if image is None:
    print("Oops! Your image is not working")

else:
    print("Image loaded successfully!")
    # cv2.circle(image, (50,50), 30, (255, 0, 0), -1)

    cv2.putText(image, "Hello Python Programmers", (30,150),
                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)


    # cv2.line(image, pt1, pt2, color, thickness)

    cv2.imshow("adding text over image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
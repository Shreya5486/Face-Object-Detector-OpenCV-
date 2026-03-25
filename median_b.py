import cv2
image = cv2.imread("filtering & blurring/flower3.jpg")  

blurred = cv2.medianBlur(image, 5)

cv2.imshow("Original Image", image)
cv2.imshow("Cleaned Image", blurred)     
cv2.waitKey(0)
cv2.destroyAllWindows()
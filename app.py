import cv2
face_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # detectMultiScale- scan and detect faces in the image, returns a list of rectangles where faces are detected
    # 1.1 balance, not too strict, not too loose. 5- minimum number of neighbors (rectangles) to retain a detection
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # x = how far from left
    # y = how far from top
    # w = width of the rectangle
    # h = height of the rectangle

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
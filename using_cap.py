import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() #ret=True/False, frame=image
    if not ret:
        print("could not read frame")
        break

    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # here, we are passing waitKey as 1, which means it will wait for 1 millisecond for a key press. If the key 'q' is pressed, it will break the loop and quit the program.
        # by using ord function, we are converting the character 'q' to its corresponding ASCII value, which is what waitKey returns when a key is pressed. The bitwise AND operation with 0xFF is used to ensure that we are only checking the lower 8 bits of the return value from waitKey, which is necessary for compatibility across different platforms and OpenCV versions.
        print("Quitting...")
        break

cap.release() #it closes the webcam feed and releases the resources associated with it. This is important to free up the webcam for other applications and to ensure that the program exits cleanly.
cv2.destroyAllWindows() #it closes all the windows that were opened by OpenCV. This is important to clean up the GUI and ensure that no windows remain open after the program has finished executing.
import cv2
import numpy as np

# Motion Detection using Background Subtraction
class MotionDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
        self.prev_frame = None
        self.motion_threshold = 500  # Minimum contour area to consider as motion

    def detect_motion(self, frame):
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        motion_boxes = []

        for contour in contours:
            if cv2.contourArea(contour) > self.motion_threshold:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append((x, y, w, h))

        return motion_detected, motion_boxes, fg_mask

def main():
    cap = cv2.VideoCapture(0)
    motion_detector = MotionDetector()

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera. Please check camera permissions and connections.")
        print("Trying to reinitialize camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to initialize camera. Exiting...")
            exit(1)

    # Load face cascade for combined detection
    face_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_frontalface_default.xml")

    motion_counter = 0
    face_counter = 0
    frame_count = 0
    error_count = 0
    max_errors = 10

    while True:
        ret, frame = cap.read()

        if not ret:
            error_count += 1
            print(f"Failed to capture frame (attempt {error_count}/{max_errors})")

            if error_count >= max_errors:
                print("Too many consecutive camera errors. Please check your camera connection.")
                break

            # Try to reinitialize camera
            cap.release()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Failed to reinitialize camera. Exiting...")
                break

            continue

        # Reset error count on successful frame capture
        error_count = 0
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect motion
        motion_detected, motion_boxes, fg_mask = motion_detector.detect_motion(gray)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Draw motion detection boxes
        for (x, y, w, h) in motion_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw face detection boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Update counters
        if motion_detected:
            motion_counter += 1
        else:
            motion_counter = max(0, motion_counter - 1)

        face_counter = len(faces)

        # Display status
        status_color = (0, 255, 0) if motion_detected else (0, 0, 255)
        status_text = "Motion Detected!" if motion_detected else "No Motion"

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Faces: {face_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Motion Level: {motion_counter}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show camera status
        if error_count > 0:
            cv2.putText(frame, "Camera: RECOVERING", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Camera: OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show foreground mask in separate window
        cv2.imshow("Motion Mask", fg_mask)
        cv2.imshow("Motion & Face Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
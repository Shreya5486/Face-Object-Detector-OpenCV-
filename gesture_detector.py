import cv2
import numpy as np
import math

class GestureDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
        self.prev_frame = None

    def detect_hand(self, frame):
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

        # Gaussian blur to reduce noise
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (likely the hand)
            max_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(max_contour) > 5000:  # Minimum area threshold
                return max_contour, skin_mask

        return None, skin_mask

    def count_fingers(self, contour, frame):
        # Find convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        if defects is None:
            return 0

        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Calculate triangle sides
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # Calculate angle
            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 180 / math.pi

            if angle <= 90 and d > 10000:  # Angle threshold and depth threshold
                finger_count += 1
                cv2.circle(frame, far, 4, [0, 0, 255], -1)

        return finger_count + 1  # Add thumb

def main():
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera. Please check camera permissions and connections.")
        print("Trying to reinitialize camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to initialize camera. Exiting...")
            exit(1)

    gesture_detector = GestureDetector()

    # Load face cascade
    face_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_frontalface_default.xml")

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

        frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Detect hand gestures
        hand_contour, skin_mask = gesture_detector.detect_hand(frame)

        gesture_text = "No Gesture"
        if hand_contour is not None:
            finger_count = gesture_detector.count_fingers(hand_contour, frame)

            # Draw hand contour
            cv2.drawContours(frame, [hand_contour], 0, (255, 0, 0), 2)

            # Determine gesture based on finger count
            if finger_count == 0:
                gesture_text = "Fist"
            elif finger_count == 1:
                gesture_text = "One Finger"
            elif finger_count == 2:
                gesture_text = "Two Fingers (Peace)"
            elif finger_count == 3:
                gesture_text = "Three Fingers"
            elif finger_count == 4:
                gesture_text = "Four Fingers"
            elif finger_count == 5:
                gesture_text = "Open Hand"
            else:
                gesture_text = f"{finger_count} Fingers"

            # Draw bounding box around hand
            x, y, w, h = cv2.boundingRect(hand_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display information
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show camera status
        if error_count > 0:
            cv2.putText(frame, "Camera: RECOVERING", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Camera: OK", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show frames
        cv2.imshow("Gesture & Face Detector", frame)
        cv2.imshow("Skin Mask", skin_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2

# Load all Haar cascades
face_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_frontalface_default.xml")
profile_face_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_profileface.xml")
eye_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_eye.xml")
eye_glasses_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_eye_tree_eyeglasses.xml")
left_eye_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_lefteye_2splits.xml")
smile_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_smile.xml")
upper_body_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_upperbody.xml")
lower_body_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_lowerbody.xml")
cat_face_cascade = cv2.CascadeClassifier("face & object detection/haarcascade_frontalcatface.xml")

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera. Please check camera permissions and connections.")
    print("Trying to reinitialize camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to initialize camera. Exiting...")
        exit(1)

frame_count = 0
error_count = 0
max_errors = 10  # Allow up to 10 consecutive errors before giving up

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

        continue  # Skip this frame and try again

    # Reset error count on successful frame capture
    error_count = 0
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect frontal faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Detect profile faces
    profile_faces = profile_face_cascade.detectMultiScale(gray, 1.1, 5)

    # Detect upper bodies
    upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 3)

    # Detect lower bodies
    lower_bodies = lower_body_cascade.detectMultiScale(gray, 1.1, 3)

    # Detect cat faces (for fun!)
    cat_faces = cat_face_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw rectangles and detect features for frontal faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        eyes_glasses = eye_glasses_cascade.detectMultiScale(roi_gray, 1.1, 10)
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

        # Combine all eye detections
        all_eyes = eyes + eyes_glasses + left_eyes

        if len(all_eyes) > 0:
            cv2.putText(frame, "Eyes", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Detect smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        if len(smiles) > 0:
            cv2.putText(frame, "Smiling", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw rectangles for profile faces
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Profile", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw rectangles for upper bodies
    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Upper Body", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw rectangles for lower bodies
    for (x, y, w, h) in lower_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, "Lower Body", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw rectangles for cat faces
    for (x, y, w, h) in cat_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(frame, "Cat Face!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Display detection counts
    face_count = len(faces) + len(profile_faces)
    body_count = len(upper_bodies) + len(lower_bodies)
    cat_count = len(cat_faces)

    cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Bodies: {body_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Cats: {cat_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show camera status
    if error_count > 0:
        cv2.putText(frame, "Camera: RECOVERING", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Camera: OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Advanced Multi-Object Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 


   
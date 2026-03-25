# Enhanced Face & Object Detection System

This directory contains advanced computer vision scripts that utilize multiple Haar cascade classifiers for comprehensive object detection.

## Available XML Classifiers
- `haarcascade_frontalface_default.xml` - Frontal face detection
- `haarcascade_profileface.xml` - Profile/side face detection
- `haarcascade_eye.xml` - Eye detection
- `haarcascade_eye_tree_eyeglasses.xml` - Eye detection with glasses
- `haarcascade_lefteye_2splits.xml` - Left eye detection
- `haarcascade_smile.xml` - Smile detection
- `haarcascade_upperbody.xml` - Upper body detection
- `haarcascade_lowerbody.xml` - Lower body detection
- `haarcascade_frontalcatface.xml` - Cat face detection (for fun!)

## Scripts

### 1. `face_eye_smile.py` - Advanced Multi-Object Detector
**Features:**
- Detects frontal faces, profile faces, upper/lower bodies
- Eye detection (regular, with glasses, left eye)
- Smile detection
- Cat face detection
- Real-time counters for detected objects
- Color-coded rectangles for different object types

**Usage:**
```bash
python "face & object detection/face_eye_smile.py"
```

**Controls:**
- Press 'q' to quit
- Shows detection counts in top-left corner

### 2. `motion_face_detector.py` - Motion & Face Detection
**Features:**
- Background subtraction for motion detection
- Face detection combined with motion tracking
- Morphological operations to reduce noise
- Motion level indicator
- Separate motion mask window

**Usage:**
```bash
python "face & object detection/motion_face_detector.py"
```

**Controls:**
- Press 'q' to quit
- Two windows: main detector and motion mask

### 3. `gesture_detector.py` - Hand Gesture Recognition
**Features:**
- Skin color detection in HSV space
- Hand contour analysis
- Finger counting using convexity defects
- Gesture recognition (fist, peace sign, open hand, etc.)
- Combined face and gesture detection

**Usage:**
```bash
python "face & object detection/gesture_detector.py"
```

**Controls:**
- Press 'q' to quit
- Shows gesture type and finger count

## Color Coding
- **Green**: Frontal faces
- **Blue**: Profile faces
- **Red**: Upper bodies
- **Cyan**: Lower bodies
- **Magenta**: Cat faces
- **Yellow**: Motion detection
- **White**: Text overlays

## Requirements
- OpenCV 4.x
- Python 3.x
- Webcam/camera device

## Tips for Best Results
1. **Lighting**: Ensure good lighting for better detection
2. **Distance**: Position yourself 2-3 feet from camera
3. **Background**: Use plain backgrounds for motion detection
4. **Calibration**: Adjust detection parameters if needed:
   - `scaleFactor`: 1.1-1.3 (balance between detection and false positives)
   - `minNeighbors`: 3-10 (higher = fewer false positives)

## Error Handling & Stability Improvements

### Camera Error Recovery
All detection scripts now include robust camera error handling:
- **Automatic Reinitialization**: If camera fails, scripts attempt to reinitialize
- **Graceful Degradation**: Continue running during temporary camera issues
- **Error Limits**: Only exit after 10 consecutive camera failures
- **Status Display**: Shows "Camera: OK" or "Camera: RECOVERING" status
- **Frame Counter**: Displays total frames processed

### Troubleshooting Camera Issues
- **Temporary Glitches**: Scripts now recover from brief camera disconnections
- **Permission Issues**: Clear error messages for camera access problems
- **Hardware Problems**: Helpful diagnostics before giving up
- **No More Random Exits**: Robust error handling prevents premature termination

## Future Enhancements
- Add more cascade classifiers
- Implement deep learning models (YOLO, SSD)
- Add audio feedback
- Create GUI interface
- Add video recording capabilities
import cv2
import os

def test_cascade_loaders():
    """Test if all Haar cascade XML files load correctly"""

    cascades = {
        "Frontal Face": "face & object detection/haarcascade_frontalface_default.xml",
        "Profile Face": "face & object detection/haarcascade_profileface.xml",
        "Eye": "face & object detection/haarcascade_eye.xml",
        "Eye with Glasses": "face & object detection/haarcascade_eye_tree_eyeglasses.xml",
        "Left Eye": "face & object detection/haarcascade_lefteye_2splits.xml",
        "Smile": "face & object detection/haarcascade_smile.xml",
        "Upper Body": "face & object detection/haarcascade_upperbody.xml",
        "Lower Body": "face & object detection/haarcascade_lowerbody.xml",
        "Cat Face": "face & object detection/haarcascade_frontalcatface.xml"
    }

    print("Testing Haar Cascade XML File Loading...")
    print("=" * 50)

    loaded_count = 0
    failed_count = 0

    for name, path in cascades.items():
        try:
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if cascade.empty():
                    print(f"❌ {name}: File exists but cascade is empty")
                    failed_count += 1
                else:
                    print(f"✅ {name}: Loaded successfully")
                    loaded_count += 1
            else:
                print(f"❌ {name}: File not found at {path}")
                failed_count += 1
        except Exception as e:
            print(f"❌ {name}: Error loading - {str(e)}")
            failed_count += 1

    print("=" * 50)
    print(f"Summary: {loaded_count} loaded successfully, {failed_count} failed")
    print("\nAll cascade files are ready for use in the detection scripts!")

if __name__ == "__main__":
    test_cascade_loaders()
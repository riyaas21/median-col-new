import cv2

def test_webcam_working():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Webcam is not working"

def test_webcam_capture():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Webcam is not working"
    
    ret, frame = cap.read()
    assert ret, "Failed to capture frame from webcam"

def test_webcam_release():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Webcam is not working"
    
    cap.release()
    assert not cap.isOpened(), "Failed to release webcam"

def test_webcam_resolution():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Webcam is not working"
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    assert width > 0, "Failed to retrieve webcam resolution (width)"
    assert height > 0, "Failed to retrieve webcam resolution (height)"

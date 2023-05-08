import cv2

def test_webcam():
    cap = cv2.VideoCapture(0) # 0 is for the default webcam, change it if necessary
    assert cap.isOpened() == True # check if the camera is opened successfully
    ret, frame = cap.read() # read the first frame
    assert ret == True # check if the frame is read successfully
    cap.release() # release the camera

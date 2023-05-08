import cv2

def test_webcam():
    cap = cv2.VideoCapture('VID2.mp4') # replace with the path to your pre-recorded video file
    assert cap.isOpened() == True # check if the camera is opened successfully

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

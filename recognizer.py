import cv2
import os
import numpy as np
import cv2
import numpy as np
import streamlit as st
import tempfile
st.title("Median detection")
import winsound

frequency = 2500  # Set frequency in Hz
duration = 1000
def intersects(x1, y1, w, h, x2, y2, x3, y3):
    # Find the minimum and maximum x and y coordinates of the rectangle
    x_min = x1
    x_max = x1 + w
    y_min = y1
    y_max = y1 + h
    
    # Find the equation of the line: y = m*x + b
    m = (y3 - y2) / (x3 - x2)
    b = y2 - m*x2
    
    # Check if the rectangle's edges intersect with the line
    for x in (x_min, x_max):
        y = m*x + b
        if y_min <= y <= y_max:
            return True
    for y in (y_min, y_max):
        x = (y - b) / m
        if x_min <= x <= x_max:
            return True
    
    # If none of the edges intersect with the line, return False
    return False


# Load video
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# If the user has uploaded a video file, save it to the current working directory
if video_file is not None:
    video_bytes = video_file.read()
    video_filename = "uploaded_video.mp4"  # Set the name of the saved file here
    with open(os.path.join(os.getcwd(), video_filename), "wb") as f:
        f.write(video_bytes)


cap = cv2.VideoCapture(video_filename)
line1_thickness = st.sidebar.slider("Line 1 Thickness", 1, 10, 2)
line1_pos = st.sidebar.slider("Line 1 Position", 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 143)
line1_tilt = st.sidebar.slider("Line 1 Tilt", -90, 90, -37)
line2_thickness = st.sidebar.slider("Line 2 Thickness", 1, 10, 2)
line2_pos = st.sidebar.slider("Line 2 Position", 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),381)
line2_tilt = st.sidebar.slider("Line 2 Tilt", -90, 90, 34)
ret, frame = cap.read()

if ret:
    line1_start = (line1_pos - line1_thickness//2, 0)
    line1_end = (line1_pos + line1_thickness//2, frame.shape[0])
    line1_rotation_matrix = cv2.getRotationMatrix2D((line1_pos, frame.shape[0]//2), line1_tilt, 1)
    line1_start = np.dot(line1_rotation_matrix, (*line1_start, 1)).astype(int)[:2]
    line1_end = np.dot(line1_rotation_matrix, (*line1_end, 1)).astype(int)[:2]
    frame = cv2.line(frame, tuple(line1_start), tuple(line1_end), (255, 0, 0), line1_thickness)

    # Draw second line
    line2_start = (line2_pos - line2_thickness//2, 0)
    line2_end = (line2_pos + line2_thickness//2, frame.shape[0])
    line2_rotation_matrix = cv2.getRotationMatrix2D((line2_pos, frame.shape[0]//2), line2_tilt, 1)
    line2_start = np.dot(line2_rotation_matrix, (*line2_start, 1)).astype(int)[:2]
    line2_end = np.dot(line2_rotation_matrix, (*line2_end, 1)).astype(int)[:2]
    frame = cv2.line(frame, tuple(line2_start), tuple(line2_end), (0, 0, 255), line2_thickness)
    # Display first frame
    st.image(frame, channels="BGR")
else:
    st.error("Failed to read video")
cap.release()
if st.button('TRAIN'):
# Define the path to the directory containing the training images
    training_dir = "C:/Users/Azmi Sharaf/Desktop/road_median/car"

    # Create an LBPH face recognizer object for training the car detection
    recognizer = cv2.face_LBPHFaceRecognizer.create()


    # Load the training images and labels
    training_images = []
    labels = []
    label_dict = {}
    current_label = 0

    for subdir in os.listdir(training_dir):
        subpath = os.path.join(training_dir, subdir)
        if os.path.isdir(subpath):
            label_dict[current_label] = subdir
            for filename in os.listdir(subpath):
                imgpath = os.path.join(subpath, filename)
                img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
                training_images.append(img)
                labels.append(current_label)
            current_label += 1

    # Train the face recognizer with the training images and labels
    recognizer.train(training_images, np.array(labels))
    recognizer.save("lbph_trained_model.yml")

    
    
       
# Load the image to be recognized

if st.button('DETECT'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("lbph_trained_model.yml")
    training_dir = "C:/Users/Azmi Sharaf/Desktop/road_median/car"
    training_images = []
    labels = []
    label_dict = {}
    current_label = 0

    for subdir in os.listdir(training_dir):
        subpath = os.path.join(training_dir, subdir)
        if os.path.isdir(subpath):
            label_dict[current_label] = subdir
            for filename in os.listdir(subpath):
                imgpath = os.path.join(subpath, filename)
                img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
                training_images.append(img)
                labels.append(current_label)
            current_label += 1
# Convert the image to graysca

    face_cascade = cv2.CascadeClassifier("C:/Users/Azmi Sharaf/Desktop/road_median/cars.xml")


    cap = cv2.VideoCapture(video_filename)
    while cap.isOpened():
    # Read the frame from the webcam
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the face cascade classifier to detect car in the frame
        cars = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        line1_start =line1_start
        line1_end = line1_end
        line1_rotation_matrix = line1_rotation_matrix
        line1_start = line1_start
        line1_end = line1_end 
        frame = cv2.line(frame, tuple(line1_start), tuple(line1_end), (255, 0, 0), line1_thickness)

        # Draw second line
        line2_start = line2_start
        line2_end = line2_end
        line2_rotation_matrix = line2_rotation_matrix
        line2_start = line2_start
        line2_end = line2_end 
        frame = cv2.line(frame, tuple(line2_start), tuple(line2_end), (0, 0, 255), line2_thickness)
        # Loop through each car in the frame
        for (x, y, w, h) in cars:
            # Draw a rectangle around the car
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the car region from the frame
            car = gray[y:y + h, x:x + w]

            # Use the face recognizer to predict the label of the car
            label, confidence = recognizer.predict(car)

            # Print the predicted label and the confidence score
            print("Predicted label:", label_dict[label])
            print("Confidence:", confidence)

            # Draw the predicted label and the confidence score above the rectangle
            text = label_dict[label] + " ({:.2f}%)".format(100 - confidence)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if intersects(x, y, w, h, line1_start[0], line1_start[1], line1_end[0], line1_end[1]) or intersects(x, y, w, h, line2_start[0], line2_start[1], line2_end[0], line2_end[1]):
                 winsound.Beep(frequency, duration)


        # Show the frame with the detected cars and labels
        cv2.imshow("Frame", frame)

        # If the 'q' key is pressed, break out of the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
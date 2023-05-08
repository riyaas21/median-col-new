import cv2
import numpy as np
import streamlit as st
import tempfile

st.set_page_config(page_title="Video Line Drawing App")

# Load video
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
if video_file is not None:
    video_bytes = video_file.read()

    # Write video bytes to temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        video_filename = f.name

    cap = cv2.VideoCapture(video_filename)



    # Set initial parameters
    line1_thickness = st.sidebar.slider("Line 1 Thickness", 1, 10, 3)
    line1_pos = st.sidebar.slider("Line 1 Position", 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2)
    line1_tilt = st.sidebar.slider("Line 1 Tilt", -90, 90, 0)
    line2_thickness = st.sidebar.slider("Line 2 Thickness", 1, 10, 3)
    line2_pos = st.sidebar.slider("Line 2 Position", 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2)
    line2_tilt = st.sidebar.slider("Line 2 Tilt", -90, 90, 0)
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
    cap = cv2.VideoCapture(video_filename)
if st.button('Click me'):
# Process video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Draw first line
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

        # Display frame with lines
        cv2.imshow("Video with Lines", frame)

        # Wait for key press to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()

    # Close window
    cv2.destroyAllWindows()

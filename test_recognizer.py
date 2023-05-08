import cv2
import os
import numpy as np
import streamlit as st
import tempfile
import winsound
import pytest

# Define the intersects function test cases
@pytest.mark.parametrize("x1, y1, w, h, x2, y2, x3, y3, expected", [
    (0, 0, 10, 10, 5, 5, 0, 0, True),  # Two rectangles intersecting
    (0, 0, 10, 10, 20, 20, 0, 0, False),  # Two rectangles not intersecting
    (0, 0, 10, 10, 5, 5, 10, 10, False),  # Two rectangles sharing a corner
    (0, 0, 10, 10, 0, 5, 10, 5, True),  # Line intersects rectangle
    (0, 0, 10, 10, 0, 5, 10, 15, False),  # Line not intersecting rectangle
])
def test_intersects(x1, y1, w, h, x2, y2, x3, y3, expected):
    assert intersects(x1, y1, w, h, x2, y2, x3, y3) == expected

# Define the test_detection function
def test_detection(video_filename):
    # Test video file upload
    video_file = open(video_filename, "rb")
    assert video_file is not None

    # Test video capture
    cap = cv2.VideoCapture(video_filename)
    ret, frame = cap.read()
    assert ret == True

    # Test line positions
    assert line1_pos >= 0 and line1_pos <= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    assert line2_pos >= 0 and line2_pos <= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Test line thicknesses
    assert line1_thickness >= 1 and line1_thickness <= 10
    assert line2_thickness >= 1 and line2_thickness <= 10

    # Test line tilts
    assert line1_tilt >= -90 and line1_tilt <= 90
    assert line2_tilt >= -90 and line2_tilt <= 90

    # Test LBPH face recognizer
    training_dir = "C:/Users/Riyaas/Desktop/road_median/car"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("lbph_trained_model.yml")
    assert recognizer is not None

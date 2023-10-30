# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:00:21 2023

@author: thanh
"""

import cv2
import face_recognition
# Load the pre-trained face detection model from OpenCV


# Open the video file
video_path = r'E:\Projet 2023\Data\Video\23-03-2023-session-1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_rate = (cap.get(cv2.CAP_PROP_FPS))
print(frame_rate)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Function to skip video to a specific frame
def skip_to_frame(video_capture, target_frame):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

# Get user input for the time to skip to (hours, minutes, seconds)
hours = int(input("Enter hours: "))
minutes = int(input("Enter minutes: "))
seconds = int(input("Enter seconds: "))

# Calculate the target frame based on the desired time
target_frame = (hours * 3600 + minutes * 60 + seconds) * frame_rate

# Skip to the target frame
skip_to_frame(cap, target_frame)

# Read the frame at the specified frame
ret, frame = cap.read()

if ret:
    # Convert the frame to RGB format (required by face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame using VGGFace
    face_locations = face_recognition.face_locations(rgb_frame)

    # Draw rectangles around detected faces
    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)
    cv2.waitKey(0)

else:
    print("Error reading frame at the specified time.")

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
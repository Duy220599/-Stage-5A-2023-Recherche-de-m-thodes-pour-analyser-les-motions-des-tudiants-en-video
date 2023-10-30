# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:36:30 2023

@author: thanh
"""

import cv2
import dlib
import os
# Load the video
video_path = r'E:\Projet 2023\Data\Video\23-03-2023-session-1.mp4'

cap = cv2.VideoCapture(video_path)
# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Print the fps
print(f"Frames per second (fps): {fps}")
# Set the starting point to 60 seconds (adjust as needed)
# Get the original frames per second (fps) of the video
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the frame increment value based on the desired fps
target_fps = 30
frame_increment = int(round(original_fps / target_fps))
print(frame_increment)
# Set the starting point to 60 seconds (adjust as needed)
start_time = 60
start_frame = int(start_time * original_fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Initialize a counter for the extracted frames
frame_counter = 0

# Directory to save the extracted face images
output_dir =  r'E:\Projet 2023\Data\Faces_data\Micheal\23-03-2023-session-1'
# os.makedirs(output_dir, exist_ok=True)

# Process the video frames
while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()

    # Check if frame reading was successful
    if not ret:
        break

    # Increment the frame counter
    frame_counter += 1

    # Extract faces every 15 frames (2 images per second)
    if frame_counter % frame_increment == 0:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detector(gray)

        # Extract the second face (if available)
        if len(faces) >= 2:
            x, y, w, h = faces[1].left(), faces[1].top(), faces[1].width(), faces[1].height()

            # Extract the face region from the frame
            face_image = frame[y:y + h, x:x + w]

            # Generate a unique filename for each extracted face
            filename = f'face_{frame_counter}.jpg'

            # Save the extracted face image to the output directory
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, face_image)
            print(f'Saved {filename}')

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
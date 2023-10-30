# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:22:24 2023

@author: thanh
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras_facenet import FaceNet
from PIL import Image, ImageDraw
import face_recognition
import time
import os
# Load the FaceNet model
embedder = FaceNet()
# Initialize the Haar cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
# Open the video file
cap = cv2.VideoCapture(r'E:\Projet 2023\Data\Video\23-03-2023-session-1.mp4')
output_dir = r'E:\Projet 2023\Data\Faces_data\Micheal\23-03-2023-session-1'
# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 2)
print(frame_interval)
# Print the fps
print(f"Frames per second (fps): {fps}")
# # Load and preprocess the reference images
reference_images = [
    # cv2.cvtColor(cv2.imread(r"E:\Projet 2023\Data\Faces_data\Reference\Micheal1.jpg"), cv2.COLOR_BGR2RGB),
    # cv2.cvtColor(cv2.imread(r"E:\Projet 2023\Data\Faces_data\Reference\Micheal2.jpg"), cv2.COLOR_BGR2RGB),
    cv2.cvtColor(cv2.imread(r"E:\Projet 2023\Data\Faces_data\Reference\Micheal3.jpg"), cv2.COLOR_BGR2RGB)
]

reference_encodings = []
for image in reference_images:
    encoding = face_recognition.face_encodings(image)
    if len(encoding) > 0:
        reference_encodings.append(encoding[0])
    else:
        print("No face detected in the reference image.")
        
# Check if any reference encodings were found
if len(reference_encodings) == 0:
    print("No valid reference encodings found. Make sure reference images contain detectable faces.")
    exit()

# Specify the start time in seconds
start_time = 100# Start at 60 seconds
# Initialize the frame counter and second counter
frame_counter = 0
seconds_counter = 0
# Define the number of frames to save per second
frames_per_second = 2
frame_number = 0
frames_saved = 0
# Calculate the frame rate and the frame index to start from
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_time * fps)
# Initialize the variables for tracking seconds and saved images
elapsed_seconds = start_time
# # Define the similarity threshold
# threshold = 0.8
# print(1)
# Skip frames until the desired starting point
for _ in range(start_frame):
    cap.read()
    

while cap.isOpened():
    # Read the next frame
      ret, frame = cap.read()
    
      if not ret:
        break

    # Skip frames until reaching the desired frame index
      if frame_counter < start_frame:
              frame_counter += 1
              continue
      # Check if the frame counter reaches the desired frame intervals
      if frame_counter % frame_interval == 0:
          # Convert the frame to RGB
          # Convert the frame to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Check if any faces are detected
        if len(face_encodings) > 0:
            # Iterate over the detected faces
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare face encodings with reference encodings
                matches = face_recognition.compare_faces(reference_encodings, face_encoding, tolerance=0.6)

                # Check if any face is a match
                if any(matches):
                    # Find the index of the best match
                    match_index = np.argmax(matches)
                    # Draw a rectangle around the detected face
                    face_location = face_locations[match_index]
                    (top, right, bottom, left) = face_location
                    # Extract the face region from the frame

                    face_image = frame[top:bottom, left:right]
                    
                    face_image = cv2.resize(face_image,(128,128))                    # Save the frame as an image
                    frame_number = frame_counter // frame_interval + 1
                    filename = f"{frame_number}.jpg"
                    cv2.imwrite(os.path.join(output_dir, filename), face_image)
                    frames_saved += 1
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    print(f'{elapsed_seconds} have passed, {frames_saved} images saved')
      frame_counter += 1
      elapsed_seconds = frame_counter / fps             
      
        # Display the resulting frame
      cv2.imshow("Video", frame)
    
        # Break the loop if 'q' is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#     # ...

#     # Increment the frame counter
#       frame_counter += 1
#       # Calculate the elapsed time in seconds
#       elapsed_seconds = frame_counter / fps

#     # Print the number of seconds and images saved
#       print(f"Elapsed seconds: {elapsed_seconds:.2f}, Images saved: {frames_saved}")

#     # Convert the frame to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect faces in the frame using Haar cascades
#     gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
    
# # Check if any faces are detected
#     if len(faces) > 0:
#         # Initialize variables for the best matching face
#         best_distance = 5 
#         best_match_index = 1
#         suitable_face = None
#         # Iterate over the detected faces
#         for i, (x, y, w, h) in enumerate(faces):
#             # Extract the face region
#             face_image = rgb_frame[y:y+h, x:x+w]

#             # Encode the face
#             face_encodings = face_recognition.face_encodings(face_image)
#             if len(face_encodings) > 0:
#                 # Calculate the distances to each reference encoding
#                 distances = [face_recognition.face_distance(reference_encodings, face_encoding) for face_encoding in face_encodings]
#                 # Find the index of the face with the smallest distance
#                 min_distance_index = np.argmin(distances)
#                 min_distance = distances[min_distance_index]
                
#                 # Update the best matching face if the distance is shorter
#                 if min_distance < best_distance:
#                     best_distance = min_distance
#                     best_match_index = i
#                     suitable_face = face_image

# # Check if a best match was found
#         if  suitable_face is not None:
#             # Draw a rectangle around the best matching face
#             x, y, w, h = faces[best_match_index]
#             face_image = cv2.resize(rgb_frame[y:y+h, x:x+w],(160,160))
            
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
#             # Put text "Micheal" on the rectangle
#             cv2.putText(frame, "Micheal", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#             # Check if the frame counter reaches the desired number of frames per second
#             if frame_counter % int(fps / frames_per_second) == 0:         
#                 # Generate the filename with seconds count and frame number
#                 frame_number = frame_counter // int(fps / frames_per_second) + 1
#                 filename = f"{frame_number}.jpg"
    
#                 # Save the frame as an image
#                 cv2.imwrite(os.path.join(output_dir, filename), face_image)
#                 frames_saved += 1 
#         # Calculate the elapsed time in seconds
#         elapsed_seconds = frame_counter / fps
#         # Check if a new second has passed
#         if elapsed_seconds >= seconds_counter + 1:
#               # Update the seconds counter
#               seconds_counter += 1
#               # Print the number of seconds and images saved
#               print(f"{seconds_counter} seconds have passed, {frames_saved} images saved")
      
       
                # Draw landmarks on the face using PIL
                # landmarks = face_recognition.face_landmarks(face_image)
                # draw = ImageDraw.Draw(pil_frame)
                # for landmark in landmarks:
                #     for feature_name, points in landmark.items():
                #         for point in points:
                #             draw.point((x + point[0], y + point[1]), fill=(255, 0, 0))
#     # Display the resulting frame
#     cv2.imshow("Video", frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# Release the video capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
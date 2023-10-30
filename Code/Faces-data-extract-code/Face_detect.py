# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:08:26 2023

@author: thanh
"""

import cv2
import numpy as np
import imutils
from imutils import paths
from facenet_pytorch import MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import torch
import face_recognition
# # Load the reference image
reference_image_path = 'E:\Projet 2023\Data\Faces_data\Reference\Micheal.jpg'
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Load the pre-trained Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(cascade_path)

mtcnn = MTCNN()
le = LabelEncoder()
svc = SVC()
video_path = r'E:\Projet 2023\Data\Video\30-03-2023 session 2.mp4'
cap = cv2.VideoCapture(video_path)


#Determine period of testig
#For testing
# Specify the start time in seconds
start_time = 100  # Start at 10 seconds

# Calculate the frame rate and the frame index to start from
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_time * fps)

# Define the similarity threshold
threshold = 0.8

# Skip frames until the desired starting point
for _ in range(start_frame):
    cap.read()
# Initialize the frame counter
frame_counter = 0

while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Process frames at the desired FPS
    # if frame_counter % 15 == 0:  # 15 frames at 30 fps to capture 2 images per second
        
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame using Haar cascades
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Initialize variables for storing the face with the highest similarity
    best_face_distance = 1.0
    best_face_location = None
    
    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_image = frame[y:y+h, x:x+w]
        
        # Resize the face image for face comparison
        face_image = cv2.resize(face_image, (150, 150))
        
        # Convert the face image to RGB (face_recognition library expects RGB format)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
       # Encode the face image (if at least one face is detected)
        face_encodings = face_recognition.face_encodings(face_image)
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            
            # Compare the face encoding with the reference encoding
            face_distance = face_recognition.face_distance([reference_encoding], face_encoding)
            
            # Check if the face similarity is below the threshold and better than the previous face
            if face_distance < threshold and face_distance < best_face_distance:
                best_face_distance = face_distance
                best_face_location = (x, y, w, h)
     
        # If a face with high similarity is found, draw a rectangle and place text on top
        if best_face_location is not None:
            x, y, w, h = best_face_location
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw green rectangle
            cv2.putText(frame, "Michael", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    frame_counter += 1
    
    # Display the resulting frame
    cv2.imshow("Video", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
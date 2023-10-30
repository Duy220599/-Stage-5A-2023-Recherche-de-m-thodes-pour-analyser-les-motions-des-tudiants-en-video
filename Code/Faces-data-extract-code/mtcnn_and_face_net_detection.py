# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:59:59 2023

@author: thanh
"""

import cv2
import numpy as np
from keras_facenet import FaceNet
import face_recognition
from mtcnn.mtcnn import MTCNN
import os
from numba import  cuda
devices = cuda.list_devices()
# Print the device information
for i, device in enumerate(devices):
    print(f"Device {i}: {device.name}")
cuda.select_device(0)

# Print the selected device information
device = cuda.get_current_device()
print(f"Selected GPU: {device.name}")
# Load the FaceNet model
embedder = FaceNet()

# Initialize the MTCNN model for face detection
detector = MTCNN()

# Open the video file
cap = cv2.VideoCapture(r'E:\Projet 2023\Data\Video\23-03-2023-session-2.mp4')
output_dir = r'E:\Projet 2023\Data\Faces_data\Micheal\23-03-2023-FaceNet'

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 2)

# Print the fps
print(f"Frames per second (fps): {fps}")

# Skip frames until the desired starting point
start_time = 0
start_frame = int(start_time * fps)
for _ in range(start_frame):
    cap.read()

frame_counter = 0
elapsed_seconds = start_time
frames_saved = 0

# Load and preprocess the reference images
reference_paths = [
    r"E:\Projet 2023\Data\Faces_data\Reference_Images\Micheal1.jpg",
    r"E:\Projet 2023\Data\Faces_data\Reference_Images\Micheal2.jpg",
    r"E:\Projet 2023\Data\Faces_data\Reference_Images\Micheal3.jpg"
]

reference_encodings = []

for path in reference_paths:
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) > 0:
        encoding = face_recognition.face_encodings(image, face_locations)[0]
        reference_encodings.append(encoding)
    else:
        print(f"No face detected in the reference image: {path}")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    if frame_counter < start_frame:
        frame_counter += 1
        continue
    
    if frame_counter % frame_interval == 0:
        # Convert the frame to RGB format (required by MTCNN)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MTCNN
        results = detector.detect_faces(rgb_frame)
        print("There are ",len(results),' faces detected')
        if len(results) > 0:
            for result in results:
                x, y, w, h = result['box']
                face = cv2.resize(rgb_frame[y:y+h, x:x+w], (160, 160))
                
                # Encode the detected face
                detected_face_encodings = face_recognition.face_encodings(face)
                if len(detected_face_encodings) > 0:
                    detected_face_encoding = detected_face_encodings[0]
                    # Compare face encodings with reference encodings
                    matches = face_recognition.compare_faces(reference_encodings, detected_face_encoding, tolerance=0.7)
                    print(matches)
                    if not any(matches):
                        # Calculate cosine distance with reference encodings and choose the closest match
                        distances = [np.linalg.norm(detected_face_encoding - ref_encoding) for ref_encoding in reference_encodings]
                        print(distances)
                        closest_match_index = np.argmin(distances)
                        if distances[closest_match_index] < 0.7:
                            matches[closest_match_index] = True
                    if any(matches):
                        print("There is 1 match")
                        face_image = cv2.resize(face, (128, 128))
                        frame_number = frame_counter // frame_interval + 1
                        minutes = int(elapsed_seconds // 60)
                        seconds = int(elapsed_seconds % 60)
                        filename = f"min_{minutes:02d}_second_{seconds:02d}_frame_{frame_number:04d}.jpg"
                        cv2.imwrite(os.path.join(output_dir, filename), face_image)
                        frames_saved += 1
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        print(f'{minutes:02d}:{seconds:02d} have passed, {frames_saved} images saved')
                    
    
    frame_counter += 1
    elapsed_seconds = frame_counter / fps
    
    # cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()


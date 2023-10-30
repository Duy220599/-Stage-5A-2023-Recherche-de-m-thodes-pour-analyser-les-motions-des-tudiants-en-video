# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:15:55 2023

@author: thanh
"""

from numba import jit, cuda
import numpy as np
import cv2
import dlib
import os
import numpy as np
import face_recognition
from timeit import default_timer as timer   
devices = cuda.list_devices()

# Print the device information
for i, device in enumerate(devices):
    print(f"Device {i}: {device.name}")
cuda.select_device(0)

# Print the selected device information
device = cuda.get_current_device()
print(f"Selected GPU: {device.name}")

# Path to the video file
video_path = r'E:\Projet 2023\Data\Video\23-03-2023-session-1.mp4'

# Output directory to save extracted faces
output_dir = r'E:\Projet 2023\Data\Faces_data\Micheal\23-03-2023-session-1'

# Path to the reference images directory
reference_dir = r'E:\Projet 2023\Data\Faces_data\Reference_Images'

# Load the face detection model
face_detector = dlib.get_frontal_face_detector()

# Load the face recognition model
face_recognition_model_path = r'E:\Projet 2023\Code\Faces-data-extract-code\dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model_path)

# Load the face landmark model
predictor_path = r"E:\Projet 2023\Code\Faces-data-extract-code\shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(predictor_path)

# Load the reference images
reference_images = []
reference_descriptors = []
# Load the reference images

reference_images = []
reference_descriptors = []
for filename in os.listdir(reference_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(reference_dir, filename)
        reference_image = dlib.load_rgb_image(image_path)
        reference_images.append(reference_image)

        # Detect face locations for the reference image
        face_locations = face_recognition.face_locations(reference_image)

        # Proceed if a face is detected in the reference image
        if len(face_locations) > 0:
            # Convert the face location to a dlib.rectangle object
            dlib_face_location = dlib.rectangle(
                face_locations[0][3], face_locations[0][0], face_locations[0][1], face_locations[0][2]
            )

            # Detect landmarks for the reference face
            reference_landmarks = landmark_predictor(reference_image, dlib_face_location)

            # Compute the face descriptor for the reference face
            reference_descriptor = np.array(face_encoder.compute_face_descriptor(reference_image, reference_landmarks))
            reference_descriptors.append(reference_descriptor)

# Load the face recognition model
face_recognition_model_path = r'E:\Projet 2023\Code\Faces-data-extract-code\dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model_path)

# Initialize variables
frame_counter = 0
frames_saved = 0
face_tracks = {}  # Dictionary to track faces across frames

# Open the video file
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Speed to catch an image
frame_interval = int(fps / 2)
time_string = 0
# Start at 60 seconds
start_time = 60
start_frame = int(start_time * fps)
elapsed_seconds = 0
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_detector(rgb_frame)

        # Check if any faces are detected
        if len(face_locations) > 0:
            # Create a new face track for each detected face
            for face_location in face_locations:
                # Generate a unique track ID
                track_id = len(face_tracks) + 1

                # Convert the face location to dlib rectangle format
                dlib_rect = dlib.rectangle(
                    face_location.left(),
                    face_location.top(),
                    face_location.right(),
                    face_location.bottom()
                )

                # Detect landmarks for the face
                landmarks = landmark_predictor(rgb_frame, dlib_rect)

                # Compute face encodings for detected face
                face_descriptor = np.array(face_encoder.compute_face_descriptor(rgb_frame, landmarks))

                # Add the face track to the dictionary
                face_tracks[track_id] = {
                    'location': dlib_rect,
                    'descriptor': face_descriptor
                }
        
                # Check if any face tracks exist
                if len(face_tracks) > 0:
                    # Create a copy of the face_tracks keys
                    track_ids = list(face_tracks.keys())
                
                    # Compute similarity scores with reference images for each face track
                    similarity_scores = []
            for track_id in track_ids:
                track_data = face_tracks[track_id]
                face_descriptor = track_data['descriptor']
                matches = face_recognition.compare_faces(reference_descriptors, face_descriptor)
                similarity_scores.append(matches)

            if len(similarity_scores) > 0:
                    if any(True in matches for matches in similarity_scores):
                        print('There are at least 1 match the image ')
                        for i, matches in enumerate(similarity_scores):
                            if True in matches:
                                best_match_track_id = track_ids[i]
                
                                # Check if the best match track ID exists in the face_tracks dictionary
                                if best_match_track_id in face_tracks:
                                    # Get the face location of the best match
                                    best_match_location = face_tracks[best_match_track_id]['location']
                
                                    # Save the face region as an image
                                    top, right, bottom, left = (
                                        best_match_location.top(),
                                        best_match_location.right(),
                                        best_match_location.bottom(),
                                        best_match_location.left()
                                    )
                                    face_image = frame[top:bottom, left:right]
                                    save_path = os.path.join(output_dir, f"face_{frames_saved}.jpg")
                                    cv2.imwrite(save_path, cv2.resize(face_image, (128, 128)))
                                    frames_saved += 1
                
                                    # Draw a rectangle around the best match face
                                    # Display the similarity score above the rectangle
                                    # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                    # text = f"Similarity: True"
                                    # cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                    # Print the elapsed time and the number of images saved
                                    print("Elapsed Time:", time_string)
                                    print("Number of Images Saved:", frames_saved)
                                    # Remove the best match face track
                                    del face_tracks[best_match_track_id]
                                    break
                    else:   
                                    min_distance = float('inf')
                                    best_match_track_id = None
                                    print('No exact match , start calculating similarities')
                                    for i, matches in enumerate(similarity_scores):
                                        track_id = track_ids[i]
                                        track_data = face_tracks.get(track_id)
                                        if track_data is not None:
                                            face_descriptor = track_data['descriptor']
                                
                                            # Calculate the distance with each reference descriptor
                                            distances = []
                                            for ref_descriptor in reference_descriptors:
                                                distance = np.linalg.norm(ref_descriptor - face_descriptor)
                                                distances.append(distance)
                                
                                            track_distance = np.mean(distances)
                                
                                            if track_distance < min_distance:
                                                best_match_track_id = track_id
                                                min_distance = track_distance
                                
                                        # Process the best match track
                                    if best_match_track_id is not None:
                                            # Get the face location of the best match
                                            best_match_location = face_tracks[best_match_track_id]['location']
                                
                                            # Save the face region as an image
                                            top, right, bottom, left = (
                                                best_match_location.top(),
                                                best_match_location.right(),
                                                best_match_location.bottom(),
                                                best_match_location.left()
                                            )
                                            face_image = frame[top:bottom, left:right]
                                            save_path = os.path.join(output_dir, f"face_{frames_saved}.jpg")
                                            cv2.imwrite(save_path, cv2.resize(face_image, (128, 128)))
                                            frames_saved += 1
                                
                                            # Draw a rectangle around the best match face
                                            # Display the similarity score above the rectangle
                                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                            text = f"Similarity: {min_distance:.2f}"
                                            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                            # Print the elapsed time and the number of images saved
                                            print("Elapsed Time:", time_string)
                                            print("Number of Images Saved:", frames_saved)
                                            # Remove the best match face track
                                            del face_tracks[best_match_track_id]
                            

    
    frame_counter += 1
    elapsed_seconds = frame_counter / fps
    # Calculate hours, minutes, and seconds
    hours = int(elapsed_seconds / 3600)
    minutes = int((elapsed_seconds % 3600) / 60)
    seconds = int(elapsed_seconds % 60)
    
    # Create the formatted time string
    time_string = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # # Display the resulting frame
    # cv2.imshow("Video", frame)
    # # Delay to keep the rectangle and similarity score visible
    # cv2.waitKey(1000)  # Adjust the delay value as needed, e.g., cv2.waitKey(1000) for a 1-second delay

    
    # if cv2.waitKey(1) == ord('q'):
    #             break

# Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()























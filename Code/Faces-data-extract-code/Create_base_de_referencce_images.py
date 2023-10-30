# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 04:21:39 2023

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
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


devices = cuda.list_devices()
# Print the device information
for i, device in enumerate(devices):
    print(f"Device {i}: {device.name}")
cuda.select_device(0)

# Print the selected device information
device = cuda.get_current_device()
print(f"Selected GPU: {device.name}")

# Load pre-trained MTCNN and FaceNet models
facenet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(margin=40)

# Path to the video file
video_path = r'E:\Projet 2023\Data\Video\23-03-2023-session-2.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_minutes = int(total_frames / (fps * 60))

# Open the video file
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Output directory to save extracted faces
output_dir = r'E:\Projet 2023\Data\Faces_data\Reference_Images\Improvisation_references'

# Path to the reference images directory
reference_dir = r'E:\Projet 2023\Data\Faces_data\Reference_Images'
# Initialize variables for reference images and timestamps
reference_interval = 3  # Time interval in minutes to update reference images
reference_images = []   # List to store reference images
reference_descriptors = []
reference_intervals = []
reference_encodings = []
start_frame = 60

for filename in os.listdir(reference_dir):
    if filename.endswith('2.jpg') or filename.endswith('.png'):
        image_path = os.path.join(reference_dir, filename)
        reference_image = cv2.imread(image_path)
        reference_images.append(reference_image)

        # Detect faces in the reference image
        reference_face_locations = face_recognition.face_locations(reference_image)
        
        if len(reference_face_locations) > 0:
            # Compute the face encoding for the reference image
            reference_encoding = face_recognition.face_encodings(reference_image, reference_face_locations)[0]
            reference_encodings.append(reference_encoding)
        else:
            print(f"No face detected in the reference image: {filename}")
        
#         reference_encodings.append(reference_encoding)
        
# # Load the face detection model
# face_detector = dlib.get_frontal_face_detector()

# # Load the face recognition model
# face_recognition_model_path = r'E:\Projet 2023\Code\Faces-data-extract-code\dlib_face_recognition_resnet_model_v1.dat'
# face_encoder = dlib.face_recognition_model_v1(face_recognition_model_path)
frame_counter = 0
frame_interval = int(fps * 60 * reference_interval)
current_interval = -1
current_minute = -1
reference_intervals = []
# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
reference_interval = 3  # Interval in minutes to extract reference faces
reference_intervals = [i for i in range(0, total_minutes + 1, reference_interval)]
print()
for minute in reference_intervals:
    target_frame = int(minute * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    print('Minute: ',minute)
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None:
            face_images = []
            face_encodings = []  # Initialize face encodings for this face
            print('There are ',len(boxes),' images detected')
            for box in boxes:
                x_left, y_top, x_right, y_bottom = [int(coord) for coord in box]
                face_image = rgb_frame[y_top:y_bottom, x_left:x_right]
                face_image = cv2.resize(face_image, (160, 160))
                face_images.append(face_image)
                
                # Detect faces in the face image
                detected_face_locations = face_recognition.face_locations(face_image)
                if len(detected_face_locations) > 0:
                    # Compute the face encoding for the detected face
                    face_encoding = face_recognition.face_encodings(face_image, detected_face_locations)[0]
                    face_encodings.append(face_encoding)
            print(len(face_encodings))
            if len(face_encodings) > 0:
                distances = np.linalg.norm(np.array(reference_encodings) - np.array(face_encodings), axis=1)
                print(distances)
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]
                print('min_distance: ',min_distance)
               #
                resized_face_image = cv2.resize(face_images[min_distance_index], (128, 128))
                save_path = os.path.join(output_dir, f"min_{minute}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(resized_face_image, cv2.COLOR_RGB2BGR))
                
                    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# cap.release()




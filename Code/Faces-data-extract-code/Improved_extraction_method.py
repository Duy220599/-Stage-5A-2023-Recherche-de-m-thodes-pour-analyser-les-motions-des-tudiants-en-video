# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:54:10 2023

@author: thanh
"""
from numba import jit, cuda
import cv2
import os
import face_recognition
import numpy as np
from facenet_pytorch import  InceptionResnetV1
from mtcnn.mtcnn import MTCNN
import torch
devices = cuda.list_devices()
# Print the device information
for i, device in enumerate(devices):
    print(f"Device {i}: {device.name}")
cuda.select_device(0)

# Print the selected device information
device = cuda.get_current_device()
print(f"Selected GPU: {device.name}")
# Initialize MTCNN and FaceNet
facenet = InceptionResnetV1(pretrained='vggface2').eval()
detector = MTCNN()
# Initialize variables
video_path = r'E:\Projet 2023\Data\Video\23-03-2023-session-2.mp4'
reference_dir = r'E:\Projet 2023\Data\Faces_data\Reference_Images\Improvisation_references'
output_dir = r'E:\Projet 2023\Data\Faces_data\Micheal\23-03-2023-session-1-improved-method'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_minutes = int(total_frames / (fps * 60))
reference_interval = 3

# Read and preprocess reference images
reference_images = []
reference_info = []
frame_counter = 0
images_saved = 0
for filename in os.listdir(reference_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(reference_dir, filename)
        reference_image = cv2.imread(image_path)
        reference_images.append(reference_image)

        reference_face = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        reference_face = torch.tensor(reference_face, dtype=torch.float32).permute(2, 0, 1) / 255
        reference_face = reference_face.unsqueeze(0)
        reference_embedding = facenet(reference_face).detach().numpy()
        current_interval = int(filename.split('_')[1].split('.')[0])
        # print('current_interval',current_interval)
        reference_info.append({'interval': current_interval, 'descriptor': reference_embedding})
print([info['interval'] for info in reference_info])
frame_counter = 0
images_saved = 0
saved_image_info = []

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Calculate current time
    elapsed_time = frame_counter / fps
    current_minute = int(elapsed_time // 60)
    current_second = int(elapsed_time % 60)
    current_interval = (current_minute // 3) * 3
    
    if frame_counter % (fps / 2) == 0 and current_interval not in [info['interval'] for info in reference_info] or current_interval == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)
        
        if len(results) > 0:
            face_images = []
            for result in results:
                x, y, w, h = result['box']
                face = cv2.resize(rgb_frame[y:y+h, x:x+w], (160, 160))
                face_images.append(face)
            
            face_descriptors = []
            for face_image in face_images:
                face_image = torch.tensor(face_image, dtype=torch.float32).permute(2, 0, 1) / 255
                face_image = face_image.unsqueeze(0)
                face_descriptor = facenet(face_image).detach()
                face_descriptors.append(face_descriptor)
            
            # Compare each face descriptor to the single reference descriptor for the interval
            matching_info_interval = [info for info in reference_info if info['interval'] == current_interval]
            best_match_distances = [np.linalg.norm(face_descriptor - info['descriptor']) for info in matching_info_interval]
            if len(best_match_distances) > 0:
                
                # Find the index of the minimum distance, which corresponds to the best match
                best_match_index = np.argmin(best_match_distances)
                print("best match index: ",best_match_index)
                if  best_match_index is not None and best_match_index < len(face_images):
                    
                    # Save matching face as a reference
                    image_info = f"min_{current_minute:02d}_second_{current_second:02d}_{images_saved:04d}.jpg"
                    save_path = os.path.join(output_dir, image_info)
                    resized_face_image = cv2.resize(face_images[best_match_index], (128, 128))
                    cv2.imwrite(save_path, cv2.cvtColor(resized_face_image, cv2.COLOR_RGB2BGR))
                    images_saved += 1
                    saved_image_info.append(image_info)

                    # Add the new reference info
                    reference_info.append({'interval': current_interval, 'descriptor': face_descriptor})

    frame_counter += 1

# Release the video capture
cap.release()
cv2.destroyAllWindows()

print("Extraction complete.")
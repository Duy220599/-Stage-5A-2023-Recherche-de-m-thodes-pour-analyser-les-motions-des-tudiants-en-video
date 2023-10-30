# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:02:36 2023

@author: thanh
"""
import shutil
import os 
import cv2
input_dir= (r'C:\Users\thanh\anaconda3\Code\Projet 2023\Data\Video')
output_dir = (r'C:\Users\thanh\anaconda3\Code\Projet 2023\Data\Faces_extracted')
frame_count = 0
frame_rate = 1

#Delete all the img exist before start
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)
    try:
        if os.path.isfile(input_dir) or os.path.islink(input_dir):
            os.unlink(input_dir)
        elif os.path.isdir(input_dir):
            shutil.rmtree(input_dir)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')
        
        
for filename in os.listdir(input_dir):
    if filename.endswith('.mp4'):
        # Open the video file
        video_file = cv2.VideoCapture(os.path.join(input_dir, filename))

        # Set the output directory for this video
        video_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        # Loop through the frames and extract every `frame_rate` frame
        frame_count = 0
        while video_file.isOpened():
            ret, frame = video_file.read()
            if not ret:
                break
            if frame_count % frame_rate == 0:
                # Part 2: Detect faces in the frame
                face_cascade = cv2.CascadeClassifier(r'C:\Users\thanh\anaconda3\Code\Projet 2023\Code\haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                # Part 3: Save each face as a separate image
                for i, (x, y, w, h) in enumerate(faces):
                    face_image = frame[y:y+h, x:x+w]
                    
                    cv2.imwrite(os.path.join(video_output_dir, f'frame_{frame_count}_face_{i}.jpg'), face_image)

                # Part 4: Display the extracted faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow('Extracted Faces', frame)
                cv2.waitKey(1)

            frame_count += frame_rate
            if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
                  
        # Release the video file
        video_file.release()

# Close all windows
cv2.destroyAllWindows()


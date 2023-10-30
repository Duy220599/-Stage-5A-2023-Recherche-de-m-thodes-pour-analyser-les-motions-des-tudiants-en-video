# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:51:03 2023

@author: thanh
"""
import dlib
import cv2
import face_recognition

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(r'E:\Projet 2023\Data\Video\22-03-2023.mp4')            
# print(1)
# person_1 = cv2.imread('img1.jpg')
# person_2 = cv2.imread('img2.jpg')
# person_3 = cv2.imread('img3.jpg')
# person_4 = cv2.imread('img4.jpg')
# person_5 = cv2.imread('img5.jpg')


# print((2))
# known_faces = [person_1,person_2,person_3,person_4,person_5]
# cv2.imshow('person1',known_faces[0])
while True:
    ret,frame = cap.read()
    
    if not ret:
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      #Detect Faces 
    cv2.imshow('Video',gray) 
    
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.5,minNeighbors = 5)
      # Match the faces with pre-trained model to detect the faces
    for (x,y,w,h) in faces:
        face_gray = gray[y:y+h,x:x+w]
        face_gray_resized = cv2.resize(face_gray,(150,150))
        regconizer =  cv2.face.LBPHFaceRegconizer_create()
        regconizer.read('trained_model.xml')
        label, confidence = regconizer.predict(face_gray_resized)
    
  # Check confidence ,if below certain threshold
        if confidence < 100:
            name = 'Person_' + str(label)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINEAA)
    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# person1 = face_recognition.load_image_file('img1.jpg')
# person2 = face_recognition.load_image_file('img2.jpg')
# person3 = face_recognition.load_image_file('img3.jpg')
# person4 = face_recognition.load_image_file('img4.jpg')
# person5 = face_recognition.load_image_file('img5.jpg')


# people = [person1, person2, person3, person4, person5]

# Create face encodings for each person's image
encodings = []
for person in people:
    encoding = face_recognition.face_encodings(person)[0]
    encodings.append(encoding)

# Load video and capture frames
cap = cv2.VideoCapture(r'E:\Projet 2023\Data\Video\16-03-2023-session-2.mp4')
print(1)
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    if not ret: # Check if the video has ended
        break
    
    # Convert the frame to RGB format for face recognition
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the RGB frame
    face_locations = face_recognition.face_locations(frame_rgb)
    
    # Loop through all detected faces in the frame
    for face_location in face_locations:
        # Extract the face region from the RGB frame
        top, right, bottom, left = face_location
        face_rgb = frame_rgb[top:bottom, left:right]
        
        # Check if a face was detected in the region
        if face_rgb.size == 0:
            continue
        
        # Resize the face image to match the size of the images of the 7 people
        face_rgb_resized = cv2.resize(face_rgb, (150, 150))
        
        # Create face encoding for the detected face
        face_encoding = face_recognition.face_encodings(face_rgb_resized)[0]
        
        # Compare the face encoding with the encodings of the 7 people using a threshold
        threshold = 0.5
        matches = face_recognition.compare_faces(encodings, face_encoding, threshold)
        
        # If there is a match, draw a rectangle around the face with the person's name
        # if True in matches:
        #     match_index = matches.index(True)
        #     name = 'Person ' + str(match_index+1)
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        #     cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Wait for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
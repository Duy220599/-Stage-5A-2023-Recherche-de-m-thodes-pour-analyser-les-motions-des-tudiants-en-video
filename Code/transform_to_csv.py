# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:25:01 2023

@author: thanh
"""

import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import csv
import sys, os
import pathlib
# Read the image and convert it to grayscale
img_path  = os.path.abspath(r'C:\Users\thanh\anaconda3\Code\Projet 2023\Code\graph.jpg')
img = cv2.imread(img_path)

if img is not None:
    # display the image
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Failed to load image:', image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




# # Threshold the image to separate the line plot from the background

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

 

# # Find the contours of the line plot

contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
 
# # Extract the XY coordinates from the contours
points = []
for cnt in contours:

    if cv2.contourArea(cnt) > 50:  # ignore small contours

        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        for pt in approx:

            points.append(pt[0])

 

# # Sort the points based on the X-axis value

points = sorted(points, key=lambda x: x[0])

 

# # Save the XY coordinates in a CSV file

with open('output.csv', 'w', newline='') as f:

    writer = csv.writer(f)

    writer.writerow(['Time', 'Pulse Rate'])

    for pt in points:

        writer.writerow([pt[0], pt[1]])
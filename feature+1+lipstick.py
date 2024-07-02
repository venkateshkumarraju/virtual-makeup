
# coding: utf-8

# In[1]:


import cv2,sys,dlib,time,math
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import faceBlendCommon as fbc

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0,8.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def lipstick(frame,faces):
    for face in faces:
    # Get the landmarks
        shape = predictor(frame, face)

        # Get the coordinates of the lips
            
        lips = []
        for i in range(48, 61):
            lips.append((shape.part(i).x, shape.part(i).y))

            # Create a mask for the lips
            mask = np.zeros_like(frame)
            cv2.fillPoly(mask, [np.array(lips)], (255, 255, 255))

            # Choose a color for the lipstick
            lipstick_color = (0,0, 255)  # BGR for red
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.convertScaleAbs(mask)

            image_lipstick = cv2.bitwise_and(frame, frame, mask=mask)
            image_lipstick[mask != 0] = lipstick_color
            # Blend the original image with the lipstick image
            frame = cv2.addWeighted(frame,1.0, image_lipstick,5, 0)
    return frame


# Open the webcam
cap = cv2.VideoCapture(0)
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_1.avi', fourcc, 10.0, (640, 480))
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    faces = detector(frame,)
    
    frame=lipstick(frame,faces )
    # Convert the image from BGR to RGB format
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    # If the frame was read successfully, display it
    if ret:
        # Write the frame into the file 'output.avi'
        out.write(frame)
        cv2.imshow('Webcam',frame)

    # If the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()



# In[ ]:





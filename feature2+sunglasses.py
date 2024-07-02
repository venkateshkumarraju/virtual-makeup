
# coding: utf-8

# In[6]:


import cv2,sys,dlib,time,math
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import faceBlendCommon as fbc

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0,8.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'

glasses = cv2.imread('sunglass.png', cv2.IMREAD_UNCHANGED)#reading the sunglass image

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

#get landmark function to dectect the face landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    if len(rects) > 0:
        return predictor(gray, rects[0])
    return None

#overlay glasses function to fit the glasses at eyes
def overlay_glasses(image, glasses, landmarks):
    # Extracting the coordinates of the eyes
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)

    # Calculate the width of the glasses based on the distance between the eyes
    eye_width = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
    glasses_width = int(eye_width * 1.6)  # Scaling factor to adjust size of the glasses

    # Calculate the center between the eyes
    center_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Calculate the top-left corner of the glasses image
    top_left = (int(center_eye[0] - glasses_width // 2), int(center_eye[1] - glasses_width // 4))

    # Resize the glasses image to fit the face
    glasses_resized = cv2.resize(glasses, (glasses_width, int(glasses_width * glasses.shape[0] / glasses.shape[1])))

    # Split the glasses image into its color and alpha channels
    gw, gh, _ = glasses_resized.shape
    glasses_rgb = glasses_resized[:, :, :3]
    glasses_alpha = glasses_resized[:, :, 3] / 255.0

    for c in range(0, 3):
        image[top_left[1]:top_left[1] + gw, top_left[0]:top_left[0] + gh, c] = (
            glasses_rgb[:, :, c] * glasses_alpha +
            image[top_left[1]:top_left[1] + gw, top_left[0]:top_left[0] + gh, c] * (1.0 - glasses_alpha)
        )

    return image

cap = cv2.VideoCapture(0)
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_2.avi', fourcc, 10.0, (640, 480))
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    faces = detector(frame,)
    
    
    landmarks = get_landmarks(frame)
    if landmarks:
    
        frame = overlay_glasses(frame, glasses,landmarks)
        
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





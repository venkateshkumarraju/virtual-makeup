# virtual-makeup

This project provides tools for applying virtual accessories to images. It includes two main features:
1. **Lipstick Feature** - This feature applies a virtual lipstick effect to an image.
2. **Sunglasses Feature** - This feature adds virtual sunglasses to an image.

This project demonstrates facial landmark detection and the implementation of various features using OpenCV, Dlib, and Matplotlib.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/venkateshkumarraju/virtual-makeup.git

# Install dependencies
pip install -r requirements.txt
```
### Usage:
Load Landmark Detector:
```
import dlib
# Landmark model location
PREDICTOR_PATH = "../resource/lib/publicdata/models/shape_predictor_68_face_landmarks.dat"
# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
```
## Read Image 
Load an image and convert it to RGB format:
```
import cv2
import matplotlib.pyplot as plt
im = cv2.imread("../resource/lib/publicdata/images/girl-no-makeup.jpg")
imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(imDlib)
```
## Calculate the facial landmarks:
```
import faceBlendCommon as fbc
points = fbc.getLandmarks(faceDetector, landmarkDetector, imDlib)
```
## Display Facial Landmarks
Display the facial landmarks on the image:
```
for i, point in enumerate(points):
    cv2.circle(imDlib, point, 2, (0, 255, 0), thickness=-1)  # Green color
    cv2.putText(imDlib, str(i), (point[0], point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)  # Red color
plt.imshow(imDlib)
```
## Apply lipstick to the detected lips:
```
import numpy as np

# Detect faces
faces = faceDetector(imDlib)

for face in faces:
    shape = landmarkDetector(imDlib, face)
    lips = [(shape.part(i).x, shape.part(i).y) for i in range(48, 61)]
    mask = np.zeros_like(imDlib)
    cv2.fillPoly(mask, [np.array(lips)], (255, 255, 255))
    lipstick_color = (0, 0, 255)  # Red color

    # Convert mask to grayscale and scale it
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.convertScaleAbs(mask)

    # Apply the lipstick
    image_lipstick = cv2.bitwise_and(imDlib, imDlib, mask=mask)
    image_lipstick[mask != 0] = lipstick_color

    # Blend the original image with the lipstick image
    imDlib = cv2.addWeighted(imDlib, 1.0, image_lipstick, 1.3, 0)

# Display the image
plt.imshow(imDlib[:, :, ::-1])
plt.show()
```
### OUTPUT 
![output_1](https://github.com/user-attachments/assets/8d433dd5-b1d2-4146-a33c-6d0aefa0208e)

## Define a function to overlay sunglasses on the detected landmarks:
```
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
```
## Real-Time Sunglasses Overlay
Capture video from the webcam and overlay sunglasses in real-time:
```
cap = cv2.VideoCapture(0)
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_2.avi', fourcc, 10.0, (640, 480))

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    faces = detector(frame)
    
    landmarks = get_landmarks(frame)
    if landmarks:
        frame = overlay_glasses(frame, glasses, landmarks)
        
    # If the frame was read successfully, display it
    if ret:
        # Write the frame into the file 'output.avi'
        out.write(frame)
        cv2.imshow('Webcam', frame)

    # If the 'q' key is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
```
### OUTPUT 
![output_2](https://github.com/user-attachments/assets/8aa78a53-c4e8-4fa0-8ecf-c07a035fdc4f)

# FAQ
## What image formats are supported?
Currently, the scripts support JPG and PNG formats.

## Can I change the color of the lipstick?
Yes, you can modify the color parameter in the feature+1+lipstick.py script to change the lipstick color.

## How do I customize the sunglasses style?
You can replace the sunglasses overlay image in the feature2+sunglasses.py script with your own image to customize the style.
## Are there any limitations?
The effectiveness of the features depends on the quality and resolution of the input images. High-quality images with clear facial features will yield better results.






# Contact
For any questions, suggestions, or feedback, please contact us at raju.venkateshkumar@gmail.com 



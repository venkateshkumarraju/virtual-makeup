# virtual-makeup
real time virtual sunglasses and lipstick
In this project i tried to build a virtual makeup application using facial landmarks. The application consists of two features:lipstick and sunglass 


**feature 1 is lipstick**
Importing necessary libraries: The script begins by importing necessary libraries such as OpenCV (cv2), dlib, numpy, matplotlib, and a custom module named faceBlendCommon.
Setting up dlib’s face detector and shape predictor: The shape_predictor object is initialized with the pre-trained model file “shape_predictor_68_face_landmarks.dat”. This model is used to detect facial landmarks in an image. The detector object is dlib’s default face detector.
Extracts the landmarks using the shape_predictor.
Gets the coordinates of the lips (landmarks 48 to 60).
Creates a mask for the lips and fills it with white color.
Defines a lipstick color in BGR format
Applies the lipstick color to the masked region in the frame.
Blends the original image with the lipstick image using cv2.addWeighted

![image](https://github.com/venkateshkumarraju/virtual-makeup/assets/160125434/beb95389-585d-4600-892e-a36df0c761c9)


**feature 2 is suglass**
Importing necessary libraries: The script begins by importing necessary libraries such as OpenCV (cv2), dlib, numpy, matplotlib, and a custom module named faceBlendCommon.
Setting up dlib’s face detector and shape predictor: The shape_predictor object is initialized with the pre-trained model file “shape_predictor_68_face_landmarks.dat”. This model is used to detect facial landmarks in an image. The detector object is dlib’s default face detector.
The coordinates of the eyes (using landmarks 36 and 45, which correspond to the left and right eyes).
The width of the glasses based on the distance between the eyes (eye_width).
The center point between the eyes (center_eye).
The top-left corner of the glasses image (top_left).
The glasses image is resized to fit into the face (glasses_resized) and splits it into color channels and an alpha channel (for transparency). blend the resize glasses image onto the original image by combining the color channels and adjusting transparency.
output 

![image](https://github.com/venkateshkumarraju/virtual-makeup/assets/160125434/0068ee33-22a9-41b1-a382-f818dbcee29b)


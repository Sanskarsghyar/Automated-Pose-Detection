import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import time
import tensorflow as tf
import keras

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

model = keras.models.load_model("weights_yoga82.best_ann.hdf5")

path = "D:\Download\Warrior-1-Pose-Virabhadrasana-1-Ekhart-Yoga.jpg"
image = cv2.imread(path)

# Initialize MediaPipe Pose
pose = mp_pose.Pose(static_image_mode=True,
                    min_detection_confidence=0.5,
                    model_complexity=2)

# Get the width and height of the frame
frame_height, frame_width, _ =  image.shape

# Resize the frame while keeping the aspect ratio.
image = cv2.resize(image, (int(frame_width * (720 / frame_height)), 720))


# Extract key points using MediaPipe Pose
results = pose.process(image)
mp_drawing.draw_landmarks(
    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Concatenate key point values into numpy array
if results.pose_landmarks is not None:
    pose_landmarks = np.array([[
            landmark.x, 
            landmark.y, 
            # landmark.z, 
            landmark.visibility
            ] 
        for landmark in results.pose_landmarks.landmark]).flatten()
else:
  pose_landmarks = np.zeros(33*3)

x = pose_landmarks.reshape(1, -1)
y_pred = model.predict(x)
label = np.argmax(y_pred)
if y_pred[0][label] > 0.75:
    if(label==0): pose_detect = 'Plank_Pose_or_Kumbhakasana_'    
    if(label==1): pose_detect = 'Warrior_I_Pose_or_Virabhadrasana_I_'  
    if(label==2): pose_detect = 'Standing_big_toe_hold_pose_or_Utthita_Padangusthasana'     
    if(label==3): pose_detect = 'Warrior_II_Pose_or_Virabhadrasana_II_'     
    if(label==4): pose_detect = 'Cat_Cow_Pose_or_Marjaryasana_'
    if(label==5): pose_detect = 'Child_Pose_or_Balasana_'  
    if(label==6): pose_detect = 'Cobra_Pose_or_Bhujangasana_'     
    if(label==7): pose_detect = 'Tree_Pose_or_Vrksasana_'     
    if(label==8): pose_detect = 'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_'

else:
    pose_detect = "No pose detected"    

cv2.putText(image, 'POSE : '+ pose_detect, (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

# cv2.imshow('dod', image)

cv2.imwrite("D:\Download\Warrior-1-Pose-Virabhadrasana-1-Ekhart-Yoga.png", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

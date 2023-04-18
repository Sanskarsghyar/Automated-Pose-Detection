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

# For webcam input:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# # For Video input
# # cap = cv2.VideoCapture("D:\Download\Code\PoseEstimation/4.mp4")

# Initialize a resizable window.
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

prevTime = 0

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
#       break
########################
    # Flip the frame horizontally for natural (selfie-view) visualization.
    image = cv2.flip(image, 1)

    # Get the width and height of the frame
    frame_height, frame_width, _ =  image.shape

    # Extract key points using MediaPipe Pose
    results = pose.process(image)
    # Draw the pose annotation on the image.
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # Numpy array of landmarks
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
    
    if y_pred[0][label] > 0.75: # Check the confidence score
        if(label==0): pose_detect = 'Plank_Pose_or_Kumbhakasana_'    
        if(label==1): pose_detect = 'Warrior_I_Pose_or_Virabhadrasana_I_'  
        if(label==2): pose_detect = 'Standing_big_toe_hold_pose_or_Utthita_Padangusthasana'     
        if(label==3): pose_detect = 'Warrior_II_Pose_or_Virabhadrasana_II_'     
        if(label==4): pose_detect = 'Cat_Cow_Pose_or_Marjaryasana_'
        if(label==5): pose_detect = 'Child_Pose_or_Balasana_'  
        if(label==6): pose_detect = 'Cobra_Pose_or_Bhujangasana_'     
        if(label==7): pose_detect = 'Tree_Pose_or_Vrksasana_'     
        if(label==8): pose_detect = 'Downward-Facing_Dog_pose_or_Adho_Mukha_Svanasana_'
        col = (255, 0, 0)
    else:
        pose_detect = "No pose detected" 
        col = (0, 0, 255)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    # cv2.putText(image, f'FPS: {int(fps)}', (1150, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 3)

    cv2.putText(image, 'POSE : ' + pose_detect, (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, col, 3)

    cv2.imshow("Pose Classification",image)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
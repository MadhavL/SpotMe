import numpy as np
import mediapipe as mp
import cv2
import utils
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# For squats we care about the following issues:

# Issues being tracked: 1) foot distance, 2) distance between knees
# Feet placement: The ratio of the distance between the subjects feet and the distance between shoulders should be within a specific range.
# Knee distance: The ratio of the distance between the subjects knees and distance between their feet should be within a specific range.


#After analyzing the "good squat" videos, here is what we got:
"""        shoulder_width  feet_width  knee_width  ratio_feet_shoulder  ratio_knee_feet
mean         0.154126    0.297969    0.210913             1.880912         0.738400
std          0.092115    0.193806    0.130554             0.341695         0.140946
min          0.067830    0.111556    0.071837             1.198069         0.526174
25%          0.087885    0.154767    0.108514             1.614046         0.607565
50%          0.098736    0.162481    0.149055             1.833358         0.725536
75%          0.242209    0.540560    0.313485             2.169355         0.855803
max          0.332358    0.616782    0.525802             2.593364         1.116864 

From this, we can set the following thresholds:
Feet-shoulder ratio: [1.2 - 2.6], Knee-feet ratio: [0.52 - 1.12]. Anything within this is good, anything outside of this is bad"""
FEET_SHOULDER_THRESHOLD = [1.2, 2.6]
KNEE_FEET_THRESHOLD = [0.53, 1.15]
error_threshold = 10

for i in range(1, 6):
    cap = cv2.VideoCapture('data/squats/test/bad' + str(i) + '.mp4')

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        error_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)

            # Extract landmarks
            if not results.pose_landmarks:
                print("No human found")
                continue
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            


            try:
                landmarks = results.pose_landmarks.landmark
                # Calculate distance between subjects shoulders
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                shoulder_width = utils.distance(left_shoulder, right_shoulder)

                # Calculate distance between subjects feet
                left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                feet_width = utils.distance(left_foot_index, right_foot_index)
                
                # Calculate distance between subjects shoulders
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                knee_width = utils.distance(left_knee, right_knee)

                # Check whether ratios are within thresholds!
                if feet_width/shoulder_width < FEET_SHOULDER_THRESHOLD[0]:
                    print("FEET DISTANCE ERROR TOO LITTLE")
                    error_count += 1

                elif feet_width/shoulder_width > FEET_SHOULDER_THRESHOLD[1]:
                    print("FEET DISTANCE ERROR TOO MUCH")
                    error_count += 1

                if knee_width/feet_width < KNEE_FEET_THRESHOLD[0]:
                    print("KNEE DISTANCE ERROR TOO LITTLE")
                    error_count += 1

                elif knee_width/feet_width > KNEE_FEET_THRESHOLD[1]:
                    print("KNEE DISTANCE ERROR TOO MUCH")
                    error_count += 1

            except:
                pass
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)
            

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    print("Error Count: " + str(error_count))
    if error_count > error_threshold:
        print("bad")
    else:
        print("good")

    cap.release()


for i in range(1, 6):

    cap = cv2.VideoCapture('data/squats/test/good' + str(i) + '.mp4')

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        error_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)

            # Extract landmarks
            if not results.pose_landmarks:
                print("No human found")
                continue
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            


            try:
                landmarks = results.pose_landmarks.landmark
                # Calculate distance between subjects shoulders
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                shoulder_width = utils.distance(left_shoulder, right_shoulder)

                # Calculate distance between subjects feet
                left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

                feet_width = utils.distance(left_foot_index, right_foot_index)
                
                # Calculate distance between subjects shoulders
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                knee_width = utils.distance(left_knee, right_knee)

                # Check whether ratios are within thresholds!
                if feet_width/shoulder_width < FEET_SHOULDER_THRESHOLD[0]:
                    print("FEET DISTANCE ERROR TOO LITTLE")
                    error_count += 1

                elif feet_width/shoulder_width > FEET_SHOULDER_THRESHOLD[1]:
                    print("FEET DISTANCE ERROR TOO MUCH")
                    error_count += 1

                if knee_width/feet_width < KNEE_FEET_THRESHOLD[0]:
                    print("KNEE DISTANCE ERROR TOO LITTLE")
                    error_count += 1

                elif knee_width/feet_width > KNEE_FEET_THRESHOLD[1]:
                    print("KNEE DISTANCE ERROR TOO MUCH")
                    error_count += 1

            except:
                pass
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)
            

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    print("Error Count: " + str(error_count))
    if error_count > error_threshold:
        print("bad")
    else:
        print("good")

    cap.release()

cv2.destroyAllWindows()
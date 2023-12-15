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

""" 
For Bicep Curls, we track the following errors:
1. Loose upper arm: ideally the upper arm limb pulling the bicep should be parallel to the torso and should not stray too far away from the body.
Can be detected by calculating the angle between the elbow, hip and the shoulder 
If the angle is greater than 10 degrees, it is an error

2. Weak peak contraction: when the forearm pulls the dumbbell upward, it may not go high enough and therefore not contract the bicep enough.
Can be detected by calculating the angle between the wrist, elbow and shoulder. 
If the minimum angle is more than 60 degrees, there is not enough contraction

"""

VISIBILITY_THRESHOLD = 0.65
UP_PHASE_THRESHOLD = 105
DOWN_PHASE_THRESHOLD = 120
CONTRACTION_THRESHOLD = 50
ALIGNMENT_THRESHOLD = 9

cap = cv2.VideoCapture(0)
stage = "down"
min_contraction_angle = 0
min_alignment_angle = 180
max_alignment_angle = -180
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
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
            right_shoulder_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            left_shoulder_landmark = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow_landmark = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist_landmark = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

            #Determine which side the curl is on, based on visibility. Then calculate the contraction and alignment angles
            if (right_shoulder_landmark.visibility > VISIBILITY_THRESHOLD and \
                right_elbow_landmark.visibility > VISIBILITY_THRESHOLD and \
                    right_wrist_landmark.visibility > VISIBILITY_THRESHOLD):
                
                #RIGHT SIDE
                right_shoulder = [right_shoulder_landmark.x, right_shoulder_landmark.y]
                right_elbow = [right_elbow_landmark.x, right_elbow_landmark.y]
                right_hip_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                right_hip = [right_hip_landmark.x, right_hip_landmark.y]
                right_wrist = [right_wrist_landmark.x, right_wrist_landmark.y]

                upper_arm_angle = utils.calculate_angle(right_hip, right_shoulder, right_elbow)
                contraction_angle = utils.calculate_angle(right_shoulder, right_elbow, right_wrist)


            elif (left_shoulder_landmark.visibility > VISIBILITY_THRESHOLD and \
                left_elbow_landmark.visibility > VISIBILITY_THRESHOLD and \
                    left_wrist_landmark.visibility > VISIBILITY_THRESHOLD):
                
                #LEFT SIDE
                left_shoulder = [left_shoulder_landmark.x, left_shoulder_landmark.y]
                left_elbow = [left_elbow_landmark.x, left_elbow_landmark.y]
                left_hip_landmark = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_hip = [left_hip_landmark.x, left_hip_landmark.y]
                left_wrist = [left_wrist_landmark.x, left_wrist_landmark.y]

                upper_arm_angle = utils.calculate_angle(left_hip, left_shoulder, left_elbow)
                contraction_angle = utils.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            if upper_arm_angle < min_alignment_angle:
                    min_alignment_angle = upper_arm_angle
            elif upper_arm_angle > max_alignment_angle:
                max_alignment_angle = upper_arm_angle

            if contraction_angle > DOWN_PHASE_THRESHOLD:
                stage = "down"
                if min_contraction_angle > CONTRACTION_THRESHOLD:
                    cv2.putText(image, "CONTRACTION ERROR", (250, 24), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, "You are not lifting the weight high enough!", (250, 48), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                

            elif contraction_angle < UP_PHASE_THRESHOLD and stage == "down":
                stage = "up"
                min_contraction_angle = 1000
                min_alignment_angle = 180
                max_alignment_angle = -180
            
            if stage == "up":
                if contraction_angle < min_contraction_angle:
                    min_contraction_angle = contraction_angle

            if max_alignment_angle - min_alignment_angle > ALIGNMENT_THRESHOLD:
                    cv2.putText(image, "ALIGNMENT ERROR", (700, 24), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, "Your arm is rotating too much around your shoulder.", (700, 48), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
            cv2.putText(image, f"{stage}", (15, 24), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

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

cap.release()
cv2.destroyAllWindows()
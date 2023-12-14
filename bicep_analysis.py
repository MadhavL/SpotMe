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

3. Lean too far back: the performer’s torso leans back or foreward during the exercise for momentum.
Due to its complexity, machine learning will be used for this detection.

"""
VISIBILITY_THRESHOLD = 0.65

DIR = 'data/bicep/good/'
videos = os.listdir(DIR)
alignment_data = []
contraction_data = []
for video in videos:
    cap = cv2.VideoCapture(DIR + video)
    contraction_angles = []
    alignment_angles = []
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
            
            try:
                landmarks = results.pose_landmarks.landmark
                right_shoulder_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                right_elbow_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                right_wrist_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                left_shoulder_landmark = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_elbow_landmark = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                left_wrist_landmark = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                #Determine which side the curl is on, based on visibility
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

                contraction_angles.append(contraction_angle)
                alignment_angles.append(upper_arm_angle)

            except:
                pass

    cap.release()
    contraction_data.append(contraction_angles)
    alignment_data.append(alignment_angles)

cv2.destroyAllWindows()

num = len(contraction_data)
rows = int(np.sqrt(num))
columns = num // rows + 1
fig, axes = plt.subplots(nrows=rows, ncols=columns)
fig2, axes2 = plt.subplots(nrows=rows, ncols=columns)
for (contraction, alignment, ax1, ax2) in zip(contraction_data, alignment_data, axes.ravel(), axes2.ravel()):
    ax1.scatter(range(len(contraction)), contraction, 4, marker='o')
    ax2.scatter(range(len(alignment)), alignment, 4, marker='o')
    ax1.set_xlabel("Frames")
    ax2.set_xlabel("Frames")
    ax1.set_ylabel("Contraction Angle")
    ax2.set_ylabel("Arm/Torso Angle")
fig.tight_layout()
fig.suptitle("Bicep Contraction Data")
fig2.suptitle("Arm Alignment Data")
fig2.tight_layout()
plt.show()
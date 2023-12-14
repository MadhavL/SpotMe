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

df = pd.DataFrame(columns=["video", "shoulder_width", "feet_width", "knee_width"])

DIR = 'data/squats/good/'
videos = os.listdir(DIR)
for video in videos:
    cap = cv2.VideoCapture(DIR + video)

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


            except:
                pass

            # Add the measurements to the dataframe
            df.loc[len(df)] = {"video": video, "shoulder_width": shoulder_width, "feet_width": feet_width, "knee_width": knee_width}


        cap.release()
cv2.destroyAllWindows()

# Calculate the ratios for the distances
df["ratio_feet_shoulder"] = df["feet_width"] / df["shoulder_width"]
df["ratio_knee_feet"] = df["knee_width"] / df["feet_width"]

print(df.describe())

# Set Theme:
sns.set_style('whitegrid')

# Creating Strip plot for feet-shoulder ratio:
sns.swarmplot(y="ratio_feet_shoulder", data=df)
plt.show()

# Creating Strip plot for knee-feet ratio:
sns.swarmplot(y="ratio_knee_feet", data=df)
plt.show()
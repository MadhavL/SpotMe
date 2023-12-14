import numpy as np
import mediapipe as mp
import cv2
import utils
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

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

GOOD = 'data/bicep/good'
BAD = 'data/bicep/bad'
videos = os.listdir(GOOD) + os.listdir(BAD)
X_train_names, X_test_names = train_test_split(videos, test_size=0.4, random_state=42)
y_train = utils.get_labels(X_train_names)
y_test = utils.get_labels(X_test_names)

# Function to extract features from videos
def load_features(videos):
    contraction_data = []
    alignment_data = []
    for video in videos:
        cap = None
        if "good" in video:
            cap = cv2.VideoCapture(GOOD + '/' + video)
        else:
            cap = cv2.VideoCapture(BAD + '/' + video)

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
        contraction_angles.extend(alignment_angles)
        contraction_data.append(contraction_angles)

    return contraction_data

def pad_lists_with_negative_ones(list_of_lists):
    # Find the maximum length of any inner list
    max_length = max(len(inner_list) for inner_list in list_of_lists)

    # Pad each inner list to the maximum length with negative ones
    padded_lists = [inner_list + [-1] * (max_length - len(inner_list)) for inner_list in list_of_lists]

    return padded_lists

X_train = load_features(X_train_names)
X_train = pad_lists_with_negative_ones(X_train)
X_train = np.array(X_train, dtype=float)

X_test = load_features(X_test_names)
X_test = pad_lists_with_negative_ones(X_test)
X_test = np.array(X_test, dtype=float)

cv2.destroyAllWindows()

knn_classifier = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = knn_classifier.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['correct', 'incorrect']))




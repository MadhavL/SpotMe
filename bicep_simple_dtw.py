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
from sklearn.metrics import classification_report

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
        contraction_data.append(contraction_angles)
        alignment_data.append(alignment_angles)

    return contraction_data, alignment_data

X_train_1, X_train_2 = load_features(X_train_names)
X_test_1, X_test_2 = load_features(X_test_names)
print(X_train_1)

cv2.destroyAllWindows()

predictions = []
print(len(X_test_names))
print(len(X_test_1))

# First Classification Approach
for example in range(len(X_test_names)):
    # Average distance to good and bad training examples
    f1_good, f1_bad, f2_good, f2_bad = [[] for i in range(4)]
    
    # Compare distance of current test example with all training examples
    for i in range(len(X_train_1)):
        dist1 = utils.DTWDistance(X_train_1[i], X_test_1[example])
        dist2 = utils.DTWDistance(X_train_2[i], X_test_2[example])
        if y_train[i]:
            f1_good.append(dist1)
            f2_good.append(dist2)
        else:
            f1_bad.append(dist1)
            f2_bad.append(dist2)

    good_score = np.mean(f1_good) + np.mean(f2_good)
    bad_score = np.mean(f1_bad) + np.mean(f2_bad)
    
    if good_score < bad_score:
        predictions.append(1)
    else:
        predictions.append(0)
    
print(classification_report(y_test, predictions, target_names=['correct', 'incorrect']))


"""
# K nearest neighbor classification
print(len(X_train))
print(len(X_test))
#print(X_train.shape)
knn = KNeighborsClassifier(n_neighbors=2, metric=utils.DTWDistance)
knn.fit(X_train, y_train)

# Evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
"""

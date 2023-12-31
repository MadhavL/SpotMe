import numpy as np
import math
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    '''
    Calculate angle between 3 points
    '''
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = radians*180.0/np.pi
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def distance(a, b):
    '''
    Calculate euclidian distance between 2 points
    '''

    x1, y1 = a
    x2, y2 = b

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def extract_specific_keypoints(results, landmark_list):
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark

    data = []
    for landmark in landmark_list:
        keypoint = landmarks[mp_pose.PoseLandmark[landmark].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()

def get_labels(array):
    '''
    Get ground truth labels (1 for good, 0 for bad)
    '''
    labels = [1 if "good" in i else 0 for i in array]
    return np.array(labels)


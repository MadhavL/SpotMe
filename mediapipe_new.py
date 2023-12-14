""" # Using newer method:

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

base_options = mp.tasks.BaseOptions
pose_landmarker = mp.tasks.vision.PoseLandmarker
pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
vision_running_mode = mp.tasks.vision.RunningMode
model_path = 'pose_landmarker_full.task'

# Create a pose landmarker instance with the video mode:
options = pose_landmarker_options(
    base_options=base_options(model_asset_path=model_path),
    running_mode=vision_running_mode.VIDEO,
    min_pose_detection_confidence=0.8,
    min_pose_presence_confidence=0.8,
    min_tracking_confidence=0.8
    )

detector = vision.PoseLandmarker.create_from_options(options)

fps = cap.get(cv2.CAP_PROP_FPS)
f = 0
while cap.isOpened():
    f += 1
    timestamp = int(1000 * f / fps)
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Process the frame with MediaPipe Pose
    pose_landmarker_result = detector.detect_for_video(mp_image, timestamp)

    if pose_landmarker_result.pose_landmarks:
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), pose_landmarker_result)
        cv2.imshow('Mediapipe Feed', annotated_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
                break """
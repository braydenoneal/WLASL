# code adapted from https://github.com/prashver/hand-landmark-recognition-using-mediapipe
import os
import cv2 as cv
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.face_mesh as mp_face
import mediapipe.python.solutions.drawing_utils as drawing
import mediapipe.python.solutions.drawing_styles as drawing_styles
import numpy as np

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5
)

face = mp_face.FaceMesh(
    static_image_mode=False,
    min_detection_confidence=0.5
)

input_dir = '../videos'
output_dir = '../mediapipe_videos'

for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)

    if os.path.isfile(file_path):
        # Open the video file
        VIDEO_FILE = file_path
        cap = cv.VideoCapture(VIDEO_FILE)

        # Get the video frame width and height
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for saving the video
        out = cv.VideoWriter(os.path.join(output_dir, filename), fourcc, 15.0, (frame_width, frame_height))

        while cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to fit the display window
            frame = cv.resize(frame, (1280, 720))  # Adjust the dimensions as needed

            # Convert the frame to RGB format
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process the frame with MediaPipe
            hands_results = hands.process(frame_rgb)
            pose_results = pose.process(frame_rgb)
            face_results = face.process(frame_rgb)

            POSE_CONNECTIONS = (
                [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                 (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)])

            # override the video with zeros to create the black background
            frame = np.zeros_like(frame)

            if pose_results.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    POSE_CONNECTIONS,
                    drawing_styles.get_default_pose_landmarks_style(),
                    is_drawing_landmarks=False
                )

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    for connections in [
                        mp_face.FACEMESH_LIPS,
                        mp_face.FACEMESH_FACE_OVAL,
                        mp_face.FACEMESH_LEFT_EYE,
                        mp_face.FACEMESH_LEFT_EYEBROW,
                        mp_face.FACEMESH_RIGHT_EYE,
                        mp_face.FACEMESH_RIGHT_EYEBROW,
                        # mp_face.FACEMESH_NOSE,
                        # mp_face.FACEMESH_TESSELATION
                    ]:
                        drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            connections,
                            is_drawing_landmarks=False
                        )

            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                        is_drawing_landmarks=False
                    )

            # Resize the video to the original size and write to a file
            frame = cv.resize(frame, (frame_width, frame_height))
            out.write(frame)

        # Release capture and output
        cap.release()
        out.release()

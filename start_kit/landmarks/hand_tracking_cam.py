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

# Set the desired resolution (e.g., 1280x720)
width, height = 1280, 720

cam = cv.VideoCapture(0)
cam.set(3, width)  # Set the width
cam.set(4, height)  # Set the height

while cam.isOpened():
    success, frame_rgb = cam.read()
    if not success:
        print("Camera Frame not available")
        continue

    # Convert image to RGB format
    frame_rgb = cv.cvtColor(frame_rgb, cv.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)
    face_results = face.process(frame_rgb)

    POSE_CONNECTIONS = frozenset([
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
        (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ])

    # Convert image to RGB format
    frame_rgb = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
    frame = np.zeros_like(frame_rgb)

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
                mp_face.FACEMESH_NOSE,
                # mp_face.FACEMESH_TESSELATION
            ]:
                drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    connections,
                    is_drawing_landmarks=False
                )

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
                is_drawing_landmarks=False
            )

    cv.imshow("Show Video", cv.flip(frame, 1))

    if cv.waitKey(20) & 0xff == ord('q'):
        break

cam.release()

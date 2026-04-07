import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initializing mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# For smoothing values
mouth_history = deque(maxlen=5)
eye_history = deque(maxlen=5)

def get_emotion(landmarks):
    # Get key points
    mouth_top = np.array([landmarks[13].x, landmarks[13].y])
    mouth_bottom = np.array([landmarks[14].x, landmarks[14].y])
    eye_top = np.array([landmarks[159].x, landmarks[159].y])
    eye_bottom = np.array([landmarks[145].x, landmarks[145].y])
    left_cheek = np.array([landmarks[234].x, landmarks[234].y])
    right_cheek = np.array([landmarks[454].x, landmarks[454].y])

    # Normalize distances by face width
    face_width = np.linalg.norm(left_cheek - right_cheek)
    mouth_open = np.linalg.norm(mouth_top - mouth_bottom) / face_width
    eye_open = np.linalg.norm(eye_top - eye_bottom) / face_width

    # Add to smoothing history
    mouth_history.append(mouth_open)
    eye_history.append(eye_open)

    mouth_avg = np.mean(mouth_history)
    eye_avg = np.mean(eye_history)

    # Rule-based emotion detection
    if mouth_avg > 0.07:
        return "Surprised 😮"
    elif eye_avg < 0.015:
        return "Angry 😠"
    elif mouth_avg > 0.04:
        return "Happy 😄"
    else:
        return "Neutral 😐"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get all landmark coordinates
            h, w, _ = frame.shape
            points = np.array([(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark])

            # Find face bounding box (rectangle)
            x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
            x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])

            # Draw the rectangle around the face
            cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

            # Get emotion and display it above the box
            emotion = get_emotion(face_landmarks.landmark)
            cv2.putText(frame, emotion, (x_min, y_min - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw landmarks (optional)
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
            )

    # Show quit message
    cv2.putText(frame, "Press 'q' to quit", (30, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

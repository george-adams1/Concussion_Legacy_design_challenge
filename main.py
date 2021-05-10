"""
Script for DesignChallenge Activity #1
Selwyn House School team George Adams, Matthew Homa, Dylan Lee
Telegram: @george_adams1

Description: Motion Tracking used to design evaluation biometric software
"""

import mediapipe as mp
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

# Setting up window
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
mp_drawing.draw_landmarks

# cap = cv2.VideoCapture(2) # 0 for integrated webcam, 2 for UVC Webcam

cap = cv2.VideoCapture('run.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'MJPG') # If using pre-recorded video
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print('Exiting...')
            break

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks) # Print results before writing

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Writing Face Object
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        # Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

        # Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # For jumping jack detection
        if results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y < results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y:
            # print('Right elbow above shoulder')
            pass
        else:
            image = cv2.putText(image, 'Put RIGHT WRIST above shoulder', (50,50), font, 1, (255,0,0), 2)
            print('Raise right elbow above shoulder')

        if results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y:
            # print('Left elbow above shoulder')
            pass
        else:
            image = cv2.putText(image, 'Put LEFT WRIST above shoulder', (50,50), font, 1, (255,0,0), 2)
            print('Raise left elbow above shoulder')


        # print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y)
        cv2.imshow('Raw Webcam Feed', image)

        # Setting up Window break on 'Q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

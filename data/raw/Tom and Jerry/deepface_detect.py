import os
import cv2
import numpy as np
from deepface import DeepFace

# Load your character info and Haar cascades
characters = [
    {
        'name': "Tom",
        'cascade': 'models/tom.xml',
        'detect_color': (165, 91, 0)
    },
    # ... add Jerry ...
]
video_path = 'videos/test/some_test_video.mp4'


def detect_haar_deepface(character, video_file):
    cap = cv2.VideoCapture(video_file)
    face_cascade = cv2.CascadeClassifier(character['cascade'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # We process every frame for this test
        # (You can add back your frame-skipping logic)

        faces = face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.10,
            minNeighbors=30,  # A medium value
            minSize=(24, 24)
        )

        for (x, y, w, h) in faces:
            roi = frame[y:y + h, x:x + w]
            emotion = "???"

            try:
                # Run emotion analysis on the region of interest
                analysis = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
            except Exception as e:
                pass  # Fails if face is too small or not "human"

            # Draw rectangles and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), character['detect_color'], 2)
            cv2.putText(frame, character['name'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow('Method 1: Haar + DeepFace', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# detect_haar_deepface(characters[0], video_path)
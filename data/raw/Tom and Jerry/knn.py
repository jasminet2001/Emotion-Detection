import os
import cv2
import numpy as np
import joblib

IMG_SIZE = (32, 32)
MODEL_PATH = 'models/knn_emotion_model.pkl'
LABELS_PATH = 'models/knn_labels.pkl'
CASCADE_PATH = 'haar_cascades/tom.xml'
VIDEO_PATH = 'videos/test/test-tom-and-jerry-1.mp4'

try:
    model = joblib.load(MODEL_PATH)
    label_names = joblib.load(LABELS_PATH)
    print("k-NN model and labels loaded.")
except FileNotFoundError:
    print(f"Error: Model files not found. Did you run train_knn.py?")
    exit()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

print("Starting detection... Press 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection and prediction
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.10,
        minNeighbors=40,
        minSize=(24, 24)
    )

    #Process Each Face Found
    for (x, y, w, h) in faces:
        # Get the face (Region of Interest)
        roi_gray = gray_frame[y:y + h, x:x + w]

        emotion = "???"
        try:
            # Process ROI exactly like training
            roi_resized = cv2.resize(roi_gray, IMG_SIZE)
            roi_flat = roi_resized.flatten().reshape(1, -1)

            # Predict emotion
            emotion = model.predict(roi_flat)[0]
        except Exception as e:
            pass  # Skip this frame if the prediction fails

        # --- 6. Draw on the *original color* frame ---
        # Draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (165, 91, 0), 2)
        cv2.putText(frame, "Tom", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #Result
    cv2.imshow('k-NN Emotion Detection (Simple)', frame)
    if cv2.waitKey(30) & 0xff == 27:
        break

print("Cleaning up...")
cv2.destroyAllWindows()
cap.release()
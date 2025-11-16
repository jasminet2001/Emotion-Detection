import os
import cv2
import numpy as np
import joblib

IMG_SIZE = (32, 32)
K_NEIGHBORS = 4
MODELS_DIR = 'models'
CASCADE_PATH = 'haar_cascades/tom.xml'
VIDEO_PATH = 'videos/test/your_test_video.mp4'

#Fix 4 things:
#1. Add a try-except block
#2. train on more videos
#3. also train for Jerry's face
#4. fix the error on this file

def euclidean_distance(img_a, img_b):
    return np.sqrt(np.sum((img_a - img_b)**2))

def prediction_scratch(X_train, y_training, test_img, k_neighbors):
    distances = []
    for img_train in X_train:
        dist = euclidean_distance(img_train, test_img)
        distances.append(dist)

    k_nearest_indices = np.argsort(distances)[:k_neighbors]
    k_nearest_labels = [y_training[i] for i in k_nearest_indices]

    #voting for the most common label
    most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
    print(most_common)
    return most_common

#Loading the model
print("Loading the model...")
model = joblib.load(os.path.join(MODELS_DIR, 'knn_data_v2.npz'))
X_data = model['X_data']
y_data = model['y_data']
label_names = joblib.load(os.path.join(MODELS_DIR, 'knn_labels.pkl'))
print(f"k-NN 'model' data loaded: {X_data.shape[0]} training samples.")

#Loading detectors
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
#Reading the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.10, minNeighbors=40, minSize=(24, 24)
    )

    #Processing Each Face Found
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        emotion = "???"

        try:
            # Process ROI
            roi_resized = cv2.resize(roi_gray, IMG_SIZE)
            roi_flat = roi_resized.flatten()

            # Predict emotion using our own function!
            emotion = prediction_scratch(X_data, y_data, roi_flat, K_NEIGHBORS)
            emotion = label_names[emotion]
            print(f"Predicted emotion: {emotion}")

        except Exception as e:
            pass  # Skip this frame in case of a prediction error


        cv2.rectangle(frame, (x, y), (x + w, y + h), (165, 91, 0), 2)
        cv2.putText(frame, "Tom", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #Results(still pretty slow)
    cv2.imshow('k-NN From Scratch', frame)

    if cv2.waitKey(1) & 0xff == 27:
        break


print("Cleaning up...")
cv2.destroyAllWindows()
cap.release()
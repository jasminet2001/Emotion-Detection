import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATA_DIR = 'emotion_dataset'
IMG_SIZE = (32, 32)

def load_data():
    images=[]
    labels=[]
    label_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    #print(label_names)

    for label in label_names:
        label_path = os.path.join(DATA_DIR, label)
        for img in os.listdir(label_path):
            img_path = os.path.join(label_path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue
            image = cv2.resize(image, IMG_SIZE)
            img_flat = image.flatten()
            images.append(np.array(img_flat))
            labels.append(np.array(label))

    return images, labels, label_names

print("Loading and processing data...")
arr_one, arr_two, label_names = load_data()

# print(f"Loaded {arr_one.shape[0]} images.")
# print(f"Features array shape: {arr_one.shape}")  # (num_images, 1024)
# print(f"Labels array shape: {arr_two.shape}")  # (num_images,)

#Testing
# Split data to see how well the model learned
X_train, X_test, y_train, y_test = train_test_split(arr_one, arr_two, test_size=0.2, random_state=42, stratify=arr_two)

print("Training k-NN model...")
# n_neighbors=5 is a good starting point.
# You can tune this number to get better accuracy.
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

#Train on ALL Data
# Now that we know it works, we train a final model on 100% of the data
print("Training final model on all data...")
final_model = KNeighborsClassifier(n_neighbors=5)
final_model.fit(arr_one, arr_two)

#Save the Model
# Save the trained model and the label names to be used in your other script
print("Saving model and labels...")
if not os.path.exists('models'):
    os.mkdir('models')

joblib.dump(final_model, 'models/knn_emotion_model.pkl')
joblib.dump(label_names, 'models/knn_labels.pkl')

print("--- Training complete! ---")

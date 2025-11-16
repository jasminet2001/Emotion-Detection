import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split  # Still useful for testing
from sklearn.metrics import accuracy_score

#in this one we implement the knn method manually
DATA_DIR = 'emotion_dataset'
IMG_SIZE = (32, 32)
MODEL_DIR = 'models'
K_NEIGHBORS = 4


def euclidean_distance(img_a, img_b):
    return np.sqrt(np.sum((img_a - img_b) ** 2))


def predict_knn(X_train, y_training, test_img, k_neighbors):
    distances = []
    for img_train in X_train:
        dist = euclidean_distance(img_train, test_img)
        distances.append(dist)

    k_nearest_indices = np.argsort(distances)[:k_neighbors]
    k_nearest_labels = [y_training[i] for i in k_nearest_indices]

    # voting for the most common label (labels are simple ints, so they are hashable)
    most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
    print(most_common)
    return most_common


def load_data():
    images = []
    labels = []

    # Get all label names (directory names) and sort them to have a stable index mapping
    all_label_names = sorted(
        [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    )

    # Map: label_name -> integer index
    label_to_index = {name: idx for idx, name in enumerate(all_label_names)}

    for label_name in all_label_names:
        label_path = os.path.join(DATA_DIR, label_name)
        label_index = label_to_index[label_name]

        for img in os.listdir(label_path):
            img_path = os.path.join(label_path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue

            image = cv2.resize(image, IMG_SIZE)
            img_flat = image.flatten().astype(np.float32)  # ensure numeric type
            images.append(img_flat)

            # Store the integer label, NOT a numpy array of a string
            labels.append(label_index)

    # Convert to numpy arrays for convenience (optional but consistent)
    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)

    return images, labels, all_label_names


print("Loading and processing data...")
X_data, y_data, label_names = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

print("Training k-NN model...")
y_pred = []
for test_img in X_test:
    pred_label = predict_knn(X_train, y_train, test_img, K_NEIGHBORS)
    y_pred.append(pred_label)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Saving the Model
print("Saving model and labels...")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# Save training data and labels for later use
np.savez(os.path.join(MODEL_DIR, 'knn_data_v2.npz'), X_data=X_data, y_data=y_data)

# Dump the label names so we can decode class indices -> label strings later
joblib.dump(label_names, os.path.join(MODEL_DIR, 'knn_labels.pkl'))

print("--- Training complete! ---")

import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATA_DIR = 'emotion_dataset/'
IMG_SIZE = (32, 32)

def load_data():
    images=[]
    labels=[]
    label_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(label_names)

load_data()
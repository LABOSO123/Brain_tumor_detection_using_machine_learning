import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load Model
model = load_model('BrainTumor_CNN.h5')

def predict_tumor(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return "Tumor Detected" if np.argmax(prediction) == 1 else "No Tumor"

# Test
image_path = "BrainTumor Classification DL/uploads/braintest.jpg"
print(f"Prediction: {predict_tumor(image_path)}")

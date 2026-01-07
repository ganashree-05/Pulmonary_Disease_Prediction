import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "pneumonia_model.h5")
CLASS_INDICES_FILE = os.path.join(MODEL_DIR, "class_indices.json")

# -----------------------------
# Load Model and Class Indices
# -----------------------------
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
if not os.path.exists(CLASS_INDICES_FILE):
    raise FileNotFoundError(f"Class indices file not found: {CLASS_INDICES_FILE}")

model = load_model(MODEL_FILE)
with open(CLASS_INDICES_FILE, "r") as f:
    class_indices = json.load(f)

print(f"✅ Model and class indices loaded: {class_indices}")

# -----------------------------
# Prediction Function
# -----------------------------
IMG_SIZE = (224, 224)


def predict_xray(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)[0][0]
    if pred >= 0.5:
        label = "PNEUMONIA"
        confidence = pred * 100
    else:
        label = "NORMAL"
        confidence = (1 - pred) * 100
    return label, confidence


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    img_path = input("Enter X-ray image path: ").strip()
    label, confidence = predict_xray(img_path)
    print(f"Prediction: {label}, Confidence: {confidence:.2f}%")

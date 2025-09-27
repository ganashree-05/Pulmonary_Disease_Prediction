import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Paths
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "pneumonia_model.h5")
CLASS_INDICES_FILE = os.path.join(MODEL_DIR, "class_indices.json")

IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# Heuristic thresholds for X-ray check
SATURATION_MEAN_THRESHOLD = 40
CHANNEL_STD_THRESHOLD = 40
GRAYSCALE_STD_MIN = 10


def load_model_and_mapping():
    if not os.path.exists(MODEL_FILE):
        print("❌ Model file not found! Run train.py first.")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_FILE)

    if os.path.exists(CLASS_INDICES_FILE):
        with open(CLASS_INDICES_FILE, "r") as f:
            class_indices = json.load(f)
        index_to_label = {int(v): k for k, v in class_indices.items()}
        print(f"✅ Loaded class indices: {class_indices}")
    else:
        index_to_label = {0: "NORMAL", 1: "PNEUMONIA"}
        print("⚠ No class_indices.json found. Using default mapping:", index_to_label)

    return model, index_to_label


def looks_like_xray(img_pil):
    img_rgb = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img_rgb)

    hsv = np.array(img_rgb.convert("HSV"))
    s = hsv[:, :, 1]
    s_mean = float(np.mean(s))

    ch_std_mean = float(np.mean(np.std(arr, axis=(0, 1))))
    gray = np.array(img_pil.convert("L").resize(IMG_SIZE))
    gray_std = float(np.std(gray))

    if s_mean > SATURATION_MEAN_THRESHOLD:
        return False
    if ch_std_mean > CHANNEL_STD_THRESHOLD:
        return False
    if gray_std < GRAYSCALE_STD_MIN:
        return False

    return True


def preprocess_for_model(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(img_path):
    model, index_to_label = load_model_and_mapping()

    if not os.path.exists(img_path):
        print("❌ Image not found:", img_path)
        return

    img_pil = Image.open(img_path)
    if not looks_like_xray(img_pil):
        print("❌ Invalid image — does not look like a chest X-ray.")
        return

    x = preprocess_for_model(img_path)
    prob = float(model.predict(x)[0][0])
    predicted_index = 1 if prob > THRESHOLD else 0

    label = index_to_label.get(predicted_index, str(predicted_index))
    confidence = prob if predicted_index == 1 else 1 - prob

    print(f"✅ Prediction: {label} (confidence {confidence:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path_to_image")
        sys.exit(1)
    img_path = sys.argv[1]
    predict_image(img_path)

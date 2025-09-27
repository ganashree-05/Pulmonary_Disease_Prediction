import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model safely
def load_model_safe(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = load_model(model_path)
        print("X-ray model loaded successfully")
        return model
    except Exception as e:
        print("Failed to load model:", e)
        return None

# Predict a single image safely
def predict_image_safe(model, img_path, target_size=(224, 224)):
    if model is None:
        raise ValueError("Model not loaded")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred_raw = model.predict(x)[0][0]
    label = "Pneumonia" if pred_raw > 0.5 else "Normal"
    confidence = round(float(pred_raw) * 100, 2)
    return label, confidence

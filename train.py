import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Paths
BASE_DIR = "data/Pneumonia_Dataset"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "pneumonia_model.h5")
CLASS_INDICES_FILE = os.path.join(MODEL_DIR, "class_indices.json")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Image size (higher resolution for accuracy)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data generators
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# Save class indices mapping
with open(CLASS_INDICES_FILE, "w") as f:
    json.dump(train_generator.class_indices, f)
print(f"✅ Saved class indices to {CLASS_INDICES_FILE}: {train_generator.class_indices}")

# Model definition (simple CNN)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# Save model
model.save(MODEL_FILE)
print(f"✅ Model saved to {MODEL_FILE}")

import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

BASE_DIR = "data/Pneumonia_Dataset"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "pneumonia_model.h5")
CLASS_INDICES_FILE = os.path.join(MODEL_DIR, "class_indices.json")

os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20      # ⭐ Increased to improve accuracy

# ---------------------------
# DATA GENERATORS
# ---------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

with open(CLASS_INDICES_FILE, "w") as f:
    json.dump(train_generator.class_indices, f)

# ---------------------------
# MODEL
# ---------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = True     # ⭐ Unfreeze last layers

# Train only last 30 layers (good for small dataset)
for layer in base_model.layers[:-30]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),   # ⭐ Smaller LR → better accuracy
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# CALLBACKS
# ---------------------------
checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)

# ---------------------------
# TRAIN
# ---------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

print("Model saved:", MODEL_FILE)

# ---------------------------
# PREDICTION
# ---------------------------
import numpy as np
from tensorflow.keras.preprocessing import image

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = "model/pneumonia_model.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

def predict_xray_model(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = MODEL.predict(img_array)[0][0]

    # high confidence output
    if pred >= 0.5:
        return "PNEUMONIA", round(pred * 100, 2)
    else:
        return "NORMAL", round((1 - pred) * 100, 2)

# -----------------------------
# Save final train & validation accuracy
# -----------------------------

# -----------------------------
# SAVE TRAINING ACCURACY DATA
# -----------------------------
# -----------------------------------------
# SAVE TRAINING ACCURACY & VALIDATION ACCURACY
# -----------------------------------------
acc_data = {
    "train_acc": float(max(history.history['accuracy'])),
    "val_acc": float(max(history.history['val_accuracy']))
}

with open(os.path.join(MODEL_DIR, "acc_data.json"), "w") as f:
    json.dump(acc_data, f)

print("✅ Training accuracy saved to acc_data.json")


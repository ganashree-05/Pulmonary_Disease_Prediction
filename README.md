🫁 Pulmonary Disease Prediction using Machine Learning & Deep Learning
📌 Project Overview

Pulmonary diseases such as Pneumonia, COPD, Tuberculosis, and Bronchitis are serious respiratory conditions that require early and accurate diagnosis. This project presents an end-to-end intelligent system that predicts pulmonary diseases using a combination of symptom-based survey analysis (Machine Learning) and chest X-ray image analysis (Deep Learning – CNN).

The system is implemented as a Flask-based web application, allowing users to input symptoms and upload chest X-ray images to receive disease predictions along with confidence scores.

🎯 Features

Symptom-based disease prediction using ML rule-based weightage

Chest X-ray image analysis using CNN (MobileNetV2)

Image preprocessing and augmentation for better accuracy

Real-time prediction with confidence percentage

User authentication (Login/Register)

SQLite database for storing user submissions

Web-based interface using Flask

🧠 Technologies Used
Programming & Frameworks

Python

Flask

TensorFlow / Keras

NumPy, OpenCV

Machine Learning & Deep Learning

CNN (Convolutional Neural Network)

Transfer Learning using MobileNetV2

Image Augmentation

Database

SQLite3

Frontend

HTML

CSS

Bootstrap

Jinja2 Templates

🧪 Dataset

Chest X-ray Dataset

Two classes:

NORMAL

PNEUMONIA

Images are in JPEG format

Preprocessed and augmented for training

⚙️ System Architecture

User Login / Registration

Symptom Survey (ML-based analysis)

X-ray Image Upload

Image Preprocessing & CNN Prediction

Disease Result & Confidence Display

Data Storage in SQLite Database

🧬 Algorithm Used
🔹 Machine Learning (Survey Analysis)

Rule-based weighted scoring system

Symptoms assigned weights based on medical importance

Final score determines disease likelihood

🔹 Deep Learning (X-ray Analysis)

Convolutional Neural Network (CNN)

Pre-trained MobileNetV2

Binary classification:

Normal

Pneumonia

🖼️ Image Preprocessing & Augmentation

Resizing images to 224×224

Normalization using preprocess_input

Data augmentation techniques:

Rotation

Zooming

Horizontal flipping

Width & height shifting

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/Pulmonary_Disease_Prediction.git
cd Pulmonary_Disease_Prediction

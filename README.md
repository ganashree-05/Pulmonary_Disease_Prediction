🫁 Pulmonary Disease Prediction using Machine Learning & Deep Learning

📌 Project Overview

Pulmonary diseases such as Pneumonia, COPD, Tuberculosis, and Bronchitis are serious respiratory conditions that require early and accurate diagnosis. This project presents an end-to-end intelligent system that predicts pulmonary diseases using a combination of symptom-based survey analysis (Machine Learning) and chest X-ray image analysis (Deep Learning – CNN).

The system is implemented as a Flask-based web application, allowing users to input symptoms and upload chest X-ray images to receive disease predictions along with confidence scores.

🎯 Features

1. Symptom-based disease prediction using ML rule-based weightage

2. Chest X-ray image analysis using CNN (MobileNetV2)

3. Image preprocessing and augmentation for better accuracy

4. Real-time prediction with confidence percentage

5. User authentication (Login/Register)

6. SQLite database for storing user submissions

7. Web-based interface using Flask

🧠 Technologies Used

1.Programming & Frameworks

Python

Flask

TensorFlow / Keras

NumPy

OpenCV

2.Machine Learning & Deep Learning

Machine Learning (Rule-based weighted scoring)

Convolutional Neural Network (CNN)

Transfer Learning using MobileNetV2

3.Database

SQLite3

4.Frontend

HTML

CSS

Bootstrap

Jinja2 Templates

🧪 Dataset

Chest X-ray Image Dataset

Two classes:

 NORMAL

 PNEUMONIA

Images are in JPEG format

Dataset is preprocessed and augmented for better generalization

⚙️ System Architecture

User Login / Registration

Symptom Survey Form

Symptom-based ML Analysis

Chest X-ray Image Upload

Image Preprocessing & CNN Prediction

Disease Result with Confidence Score

Storage of Results in SQLite Database

🧬 Algorithms Used

🔹 Machine Learning (Survey Analysis)

Rule-based weighted scoring algorithm

Each symptom is assigned a predefined medical weight

Total score determines disease likelihood

🔹 Deep Learning (X-ray Analysis)

Convolutional Neural Network (CNN)

Transfer Learning using MobileNetV2

Binary Classification:

 Normal
 
 Pneumonia

🖼️ Image Preprocessing & Augmentation

Image resizing to 224 × 224

Normalization using preprocess_input

Data augmentation techniques:

Rotation

Zooming

Horizontal flipping

Width and height shifting

🚀 How to Run the Project

Step 1: Clone the Repository

git clone https://github.com/your-username/Pulmonary_Disease_Prediction.git
cd Pulmonary_Disease_Prediction

Step 2: Install Required Packages

pip install -r requirements.txt

Step 3: Train the Model

python train.py

Step 4: Run the Flask Application

python app.py

Step 5: Open in Browser

http://127.0.0.1:5000/

🗂️ Project Structure

Pulmonary_Disease_Prediction/
│
├── app.py
├── train.py
├── predict.py
├── model/
│   └── pneumonia_model.h5
├── data/
│   └── Pneumonia_Dataset/
├── templates/
├── static/
├── database.db
├── requirements.txt
└── README.md





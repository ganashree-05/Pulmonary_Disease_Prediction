🫁 Pulmonary Disease Prediction using Machine Learning & Deep Learning

📌 Project Overview
Pulmonary diseases such as Pneumonia, COPD, Tuberculosis, and Bronchitis are serious respiratory conditions that require early and accurate diagnosis. This project presents an end-to-end intelligent system that predicts pulmonary diseases using a combination of symptom-based survey analysis (Machine Learning) and chest X-ray image analysis (Deep Learning – CNN).
The system is implemented as a Flask-based web application, allowing users to input symptoms and upload chest X-ray images to receive disease predictions along with confidence scores.

🎯 Features

1.Symptom-based disease prediction using ML rule-based weightage

2.Chest X-ray image analysis using CNN (MobileNetV2)

3.Image preprocessing and augmentation for better accuracy

4.Real-time prediction with confidence percentage

5.User authentication (Login/Register)

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

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
python train.py

4️⃣ Run the Flask Application
python app.py

5️⃣ Open in Browser
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

📊 Output

Disease prediction result
Confidence percentage
Stored user history and predictions

🔮 Future Enhancements

Multi-disease classification
Integration with real hospital systems
Support for CT scan images
Cloud deployment
Mobile application support

🤝 Contribution

This project aims to assist healthcare professionals and students by providing an intelligent decision-support system for early pulmonary disease detection.

📄 License

This project is developed for educational and research purposes.

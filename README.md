# Age and Gender Detection Using Computer Vision

This repository contains a deep learning-based application for detecting the **age** and **gender** of individuals from facial images. The project uses **Convolutional Neural Networks (CNNs)** and **OpenCV** for face detection, and a **Streamlit** web app for deployment.

---

## 🚀 Features

- 🎯 Real-time age and gender detection from webcam or uploaded images.
- 🧠 Built using TensorFlow/Keras with a custom age-scaling layer.
- 🎥 Face detection via OpenCV Haar cascades.
- 📷 Interactive frontend using Streamlit.
- 📦 Lightweight, fast, and easy to deploy on local or cloud servers.
- 🔐 No data storage — privacy-friendly design.

---

## 🧩 Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python, TensorFlow, Keras
- **Face Detection:** OpenCV
- **Model:** CNN with dual output (binary classification for gender, regression for age)

---

## 📁 Dataset

- **Dataset Used:** [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- **Size:** ~23,000 face images
- **Labels:** Age (0–116), Gender (0 = Male, 1 = Female)
- **Preprocessing:**
  - Resized to 128x128
  - Normalized pixel values to [0, 1]
  - Extracted labels from filenames

---

## 📷 How It Works

1. **Face Detection**: Uses Haar Cascade to detect faces in uploaded images or webcam feed.
2. **Preprocessing**: Detected faces are resized and normalized.
3. **Prediction**: CNN model outputs:
   - **Gender**: Male or Female
   - **Age**: Integer value from 0–100
4. **Annotation**: Draws bounding boxes and labels the face on the original image.
5. **Display**: Results shown on a Streamlit interface in real time.

---

## 🛠 Installation

```bash
git clone https://github.com/your-username/age-gender-detection.git
cd age-gender-detection
pip install -r requirements.txt
```

---

## 🧪 Run the App

```bash
streamlit run stream.py
```

- Upload an image or enable the webcam.
- View predicted gender and age instantly.

---

## 📦 Model

- **File**: `gender_age_model.h5`
- **Custom Layer**: `AgeScalingLayer` (used to scale age predictions between 0–100)
- **Architecture**:
  - 4 Convolutional layers
  - MaxPooling
  - Flatten → Dense layers with dropout
  - 2 outputs: `gender_out` (sigmoid), `age_out` (relu)

---

## 📊 Evaluation Metrics

- **Gender**: Accuracy
- **Age**: Mean Absolute Error (MAE)

Model trained on:
- **30 epochs**
- **Batch size**: 32
- **Validation split**: 20%

---

## 💡 Future Scope

- Multi-face detection support
- Mobile app deployment (Android/iOS)
- Emotion recognition
- Edge device optimization (e.g., Jetson Nano, Raspberry Pi)
- Integration with face recognition or biometric systems
- Privacy-focused enhancements

---

## 🧠 Learnings

This project provided in-depth hands-on experience with:

- CNN model building & multi-output training
- Real-time webcam inference
- Streamlit app deployment
- Ethical AI considerations (bias, fairness, privacy)

---

## 📚 References

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- [UTKFace Dataset - Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)

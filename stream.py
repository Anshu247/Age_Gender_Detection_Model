import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from PIL import Image

# Custom layer used in the trained model
class AgeScalingLayer(Layer):
    def call(self, inputs):
        return inputs * 100

# Load model once
@st.cache_resource
def load_model_once():
    return load_model(
        'gender_age_model.h5',
        custom_objects={'AgeScalingLayer': AgeScalingLayer},
        compile=False
    )

model = load_model_once()

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_dict = {0: 'Male', 1: 'Female'}

# Function to annotate image with predictions
def annotate_image(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        faces = [(0, 0, img.shape[1], img.shape[0])]

    for (x, y, w, h) in faces:
        face = img_copy[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (128, 128))
        face_array = face_resized.astype(np.float32) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        pred = model.predict(face_array, verbose=0)
        gender = gender_dict[round(pred[0][0][0])]
        age = int(min(100, max(0, round(pred[1][0][0]))))

        label = f"{gender}, {age}"
        font_scale = max(1.0, w / 150)
        thickness = 2

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        
    return img_copy

# Webcam processing function
def process_webcam():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button = st.button("Stop Camera")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        annotated = annotate_image(frame)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_rgb, channels="RGB", use_container_width=True)

    cap.release()

# Streamlit UI setup
st.set_page_config(page_title="Age & Gender Detection", layout="centered")
st.title("🧑‍🔬 Age and Gender Detection")
st.write("Upload an image or use your webcam to detect age and gender of face(s).")

# Upload image option
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Flag to determine if detection is done
detection_done = False

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read the image. Please try another file.")
    else:
        detect_button = st.button("Detect from Image")
        if detect_button:
            with st.spinner("Detecting faces and predicting..."):
                annotated = annotate_image(img)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.subheader("Prediction Result")
                st.image(annotated_rgb, use_container_width=True)
                detection_done = True
        if not detect_button:
            st.subheader("Original Image")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)

# Live camera option
st.markdown("---")
st.subheader("Or try with your Webcam")
if st.button("Start Live Camera"):
    process_webcam()
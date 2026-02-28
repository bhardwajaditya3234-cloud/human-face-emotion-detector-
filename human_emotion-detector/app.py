import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("Human Emotion Detection")
st.write("Upload a face image and the system will predict the emotion.")

# Load model
model = load_model("emotion_model.h5")

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMG_SIZE = 48

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected. Please upload a clear face image.")
    else:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"Predicted Emotion: {emotion.upper()}")
        st.info(f"Confidence: {confidence:.2f}%")

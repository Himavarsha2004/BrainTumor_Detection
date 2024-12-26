import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('braintumor.h5')

# Define the labels
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Set Streamlit page configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="centered", page_icon="🧠")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction"])

# Home Page
if page == "Home":
    # Header and introduction
    st.title("🧠 Brain Tumor Detection")
    st.subheader("About the Project")
    st.write(
        """
        This project uses a deep learning model to classify MRI scans of brain tumors 
        into four categories:
        - Glioma Tumor
        - Meningioma Tumor
        - No Tumor
        - Pituitary Tumor

        The goal is to assist medical professionals in the accurate diagnosis of brain tumors.
        """
    )

    # Technologies Used
    st.subheader("🛠️ Technologies Used")
    st.write(
        """
        - **Python**  
        - **TensorFlow/Keras**  
        - **OpenCV**  
        - **Streamlit**  
        - **Matplotlib & Seaborn**
        """
    )

    # Display model accuracy
    st.subheader("📊 Model Accuracy")
    st.metric(label="Training Accuracy", value="95.59%")
    st.metric(label="Validation Accuracy", value="90.14%")

# Prediction Page
elif page == "Prediction":
    st.title("🔍 Brain Tumor Classification")
    st.write("Upload an MRI image below to classify the type of brain tumor.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded file
        file_bytes = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Resize the image as done in the notebook
        img = cv2.resize(img, (150, 150))

        # Reshape the image
        img_array = img.reshape(1, 150, 150, 3)

        # Display the image (smaller size)
        st.image(Image.open(uploaded_file), caption='Uploaded MRI Image', width=300)

        # Make predictions using the same logic as in the notebook
        predictions = model.predict(img_array)
        predicted_index = predictions.argmax()
        predicted_label = labels[predicted_index]

        # Show the result
        st.success(f"### Prediction: **{predicted_label}**")

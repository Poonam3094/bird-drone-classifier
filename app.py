import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("bird_drone_model.keras")

st.title("?? Bird vs Drone Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("?? Drone Detected")
        st.write(f"Confidence: {prediction[0][0]:.2f}")
    else:
        st.success("?? Bird Detected")
        st.write(f"Confidence: {1 - prediction[0][0]:.2f}")

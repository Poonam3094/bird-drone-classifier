import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# Load TFLite Model
# -------------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Bird vs Drone Classifier", layout="centered")

st.title("🦅 Bird vs Drone Classifier")
st.write("Upload an aerial image to classify whether it's a **Bird 🐦** or a **Drone 🚁**.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:

    # Load and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])
    score = float(prediction[0][0])

    # Output result
    st.subheader("🔍 Prediction Result")

    if score > 0.5:
        st.success(f"🚁 Drone Detected")
        st.write(f"Confidence: {score:.2f}")
    else:
        st.success(f"🐦 Bird Detected")
        st.write(f"Confidence: {1 - score:.2f}")

    # Confidence bar
    st.progress(score if score > 0.5 else 1 - score)

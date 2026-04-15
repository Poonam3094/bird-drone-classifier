import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🦅 Bird vs Drone Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption="Uploaded Image")

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    if prediction[0][0] > 0.5:
        st.success("🚁 Drone Detected")
        st.write(f"Confidence: {prediction[0][0]:.2f}")
    else:
        st.success("🐦 Bird Detected")
        st.write(f"Confidence: {1 - prediction[0][0]:.2f}")

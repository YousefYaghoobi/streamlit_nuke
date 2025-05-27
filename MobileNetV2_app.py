import streamlit as st
st.set_page_config(page_title="ImageNet Classifier", layout="centered")  # ✅ First Streamlit command

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load pre-trained model
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

st.title("📷 Image Classifier using MobileNetV2")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=3)[0]

    st.subheader("🔍 Top Predictions:")
    for i, (imagenet_id, label, prob) in enumerate(decoded):
        st.write(f"{i+1}. **{label}** – {prob:.2%}")

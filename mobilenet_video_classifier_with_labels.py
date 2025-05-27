# -*- coding: utf-8 -*-
"""
Created on Tue May 27 03:59:18 2025

@author: zzulk
"""

import streamlit as st
st.set_page_config(page_title="Video Frame Classifier", layout="centered")

import numpy as np
import tensorflow as tf
import tempfile
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load MobileNetV2 model with ImageNet weights
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

st.title("🎥 Video Classifier with Labels using MobileNetV2")
st.markdown("Upload a video. Each frame will be classified using a pre-trained MobileNetV2 model.")

uploaded_video = st.file_uploader("Upload a video (.mp4 or .avi)", type=["mp4", "avi"])
if uploaded_video:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    frame_count = 0

    while cap.isOpened() and frame_count < 100:  # Limit to 100 frames
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame for MobileNetV2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        img_array = preprocess_input(np.expand_dims(resized, axis=0))

        # Predict
        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=1)[0][0]  # Get top prediction
        label = decoded[1]
        confidence = decoded[2]

        # Draw prediction on frame
        display_text = f"{label} ({confidence:.2%})"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        stframe.image(frame, channels="BGR", use_column_width=True)
        frame_count += 1

    cap.release()
    st.success("✅ Video classification finished!")

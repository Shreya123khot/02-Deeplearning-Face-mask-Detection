# pip install streamlit tensorflow pillow numpy pandas

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ---------------- Page ----------------
st.set_page_config(page_title="Face Mask Detection", page_icon="😷")

st.title("😷 Face Mask Detection")

# ---------------- Session State ----------------
if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

# ---------------- Load Model ----------------
try:
    model = load_model("mask_final.keras")
except Exception as e:
    st.error(f"Error loading model:\n{e}")
    st.stop()

# ---------------- Upload Image ----------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((128, 128))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    prob = prediction[0][0]

    if prob > 0.5:
        st.error(f"Prediction : WITHOUT MASK 😷\nConfidence : {prob*100:.2f}%")
    else:
        st.success(f"Prediction : WITH MASK ✅\nConfidence : {(1-prob)*100:.2f}%")

# ---------------- Camera Buttons ----------------
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("📸 Open Camera"):
        st.session_state.open_camera = True

with col2:
    if st.button("❌ Close Camera"):
        st.session_state.open_camera = False

# ---------------- Camera ----------------
if st.session_state.open_camera:

    camera_image = st.camera_input("Click Photo")

    if camera_image is not None:

        img = Image.open(camera_image).convert("RGB")
        img = img.resize((128, 128))

        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        confidence = prediction[0][0]

        if confidence > 0.5:
            st.error(f"WITHOUT MASK 😷\nConfidence : {confidence*100:.2f}%")
        else:
            st.success(f"WITH MASK ✅\nConfidence : {(1-confidence)*100:.2f}%")

        st.session_state.open_camera = False

st.markdown("---")

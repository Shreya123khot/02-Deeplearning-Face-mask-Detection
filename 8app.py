# pip install streamlit tensorflow pillow numpy pandas

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Face Mask Detection")

# ---------------- Load Model ----------------
@st.cache_resource
def load_my_model():
    return load_model("mask_final.keras", compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model Loading Error:\n{e}")
    st.stop()

# ---------------- Prediction Function ----------------
def predict_mask(img):

    img = img.convert("RGB")
    img = img.resize((128, 128))

    img = img_to_array(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    prob = float(pred[0][0])

    return prob

# ---------------- Upload ----------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    prob = predict_mask(image)

    st.progress(min(prob, 1.0))

    st.write(f"Prediction Score : {prob:.4f}")

    if prob >= 0.5:
        st.success(f"✅ WITH MASK\nConfidence : {prob*100:.2f}%")
    else:
        st.error(f"❌ WITHOUT MASK\nConfidence : {(1-prob)*100:.2f}%")

st.divider()

# ---------------- Camera ----------------
camera = st.camera_input("📸 Capture Image")

if camera:

    image = Image.open(camera)

    st.image(image, caption="Captured Image", use_container_width=True)

    prob = predict_mask(image)

    st.progress(min(prob, 1.0))

    st.write(f"Prediction Score : {prob:.4f}")

    if prob >= 0.5:
        st.success(f"✅ WITH MASK\nConfidence : {prob*100:.2f}%")
    else:
        st.error(f"❌ WITHOUT MASK\nConfidence : {(1-prob)*100:.2f}%")
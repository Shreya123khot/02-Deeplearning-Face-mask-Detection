# pip install streamlit tensorflow pillow numpy pandas

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Face Mask Detection")
st.write("Upload an image or capture a photo to detect whether a person is wearing a face mask.")

# ---------------- Session State ----------------
if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

# ---------------- Load Model ----------------
@st.cache_resource
def get_model():
    return load_model("mask_final.keras", compile=False)

try:
    model = get_model()
except Exception as e:
    st.error(f"❌ Unable to load model.\n\n{e}")
    st.stop()


# ---------------- Prediction Function ----------------
def predict_mask(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))

    img_array = image.img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    probability = float(prediction[0][0])

    return probability


# ---------------- Upload Image ----------------
uploaded_file = st.file_uploader(
    "📂 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    prob = predict_mask(img)

    st.markdown("### Prediction")

    if prob > 0.5:
        st.error(
            f"😷 WITHOUT MASK\n\nConfidence : {prob*100:.2f}%"
        )
    else:
        st.success(
            f"✅ WITH MASK\n\nConfidence : {(1-prob)*100:.2f}%"
        )

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

    camera_image = st.camera_input("Take a Photo")

    if camera_image is not None:

        img = Image.open(camera_image)

        st.image(
            img,
            caption="Captured Image",
            use_container_width=True
        )

        prob = predict_mask(img)

        st.markdown("### Prediction")

        if prob > 0.5:
            st.error(
                f"😷 WITHOUT MASK\n\nConfidence : {prob*100:.2f}%"
            )
        else:
            st.success(
                f"✅ WITH MASK\n\nConfidence : {(1-prob)*100:.2f}%"
            )

        st.session_state.open_camera = False

st.markdown("---")
st.caption("Developed using Streamlit & TensorFlow")

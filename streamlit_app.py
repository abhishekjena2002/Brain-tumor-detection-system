#%%
# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image and predict Tumor / No Tumor")

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        r"artifacts/MobileNetV2.keras"
    )

    return model


model = load_model()

# ==========================================
# PREPROCESS FUNCTION
# ==========================================

def preprocess_image(image):

    image = np.array(image)

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    # ======================================
    # PREDICTION
    # ======================================

    processed_image = preprocess_image(
        image
    )

    prediction = model.predict(
        processed_image
    )[0][0]

    tumor_probability = float(
        prediction
    )

    no_tumor_probability = (
        1 - tumor_probability
    )

    st.markdown("---")

    if tumor_probability >= 0.5:

        st.error(
            "🚨 Tumor Detected"
        )

        st.metric(
            "Tumor Probability",
            f"{tumor_probability*100:.2f}%"
        )

    else:

        st.success(
            "✅ No Tumor Detected"
        )

        st.metric(
            "No Tumor Probability",
            f"{no_tumor_probability*100:.2f}%"
        )

    st.markdown("---")

    st.subheader("Prediction Scores")

    st.write(
        f"Tumor: {tumor_probability*100:.2f}%"
    )

    st.write(
        f"No Tumor: {no_tumor_probability*100:.2f}%"
    )


# %%

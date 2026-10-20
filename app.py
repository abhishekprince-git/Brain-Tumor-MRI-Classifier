
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Model Configuration
# -----------------------------
REPO_ID = "abhishek2prince/brain-tumor-convnext"

MODEL_OPTIONS = {
    "EfficientNetB0": "efficientnet",
    "ConvNeXt-Tiny (BCE Loss)": "convnext_tiny_bce.keras",
    "ConvNeXt-Tiny (Focal Loss)": "convnext_tiny_focal.keras",
}

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_local_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )

@st.cache_resource
def load_huggingface_model(filename):
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model"
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )

# -----------------------------
# Prediction Function
# -----------------------------
def predict_mri(image, model):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Notebook uses RGB float32 values in the 0-255 range.
    image_array = np.expand_dims(image_array, axis=0)

    output = model.predict(image_array, verbose=0)
    tumour_probability = float(np.asarray(output).reshape(-1)[0])

    # Binary mapping verified from the ConvNeXt notebook:
    # 0 = Non-Tumour, 1 = Tumour
    prediction = (
        "Tumour"
        if tumour_probability >= 0.5
        else "Non-Tumour"
    )

    class_probability = (
        tumour_probability
        if prediction == "Tumour"
        else 1.0 - tumour_probability
    )

    return prediction, class_probability * 100, tumour_probability

# -----------------------------
# User Interface
# -----------------------------
st.title("🧠 Brain Tumor MRI Classifier")

st.write(
    "Upload a brain MRI image and select a trained model "
    "to obtain a binary tumour classification."
)

selected_model = st.selectbox(
    "Select Model",
    list(MODEL_OPTIONS.keys())
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded MRI",
            use_container_width=True
        )

        if st.button("🔍 Predict", type="primary"):
            with st.spinner("Loading model and analysing image..."):
                model_choice = MODEL_OPTIONS[selected_model]

                if model_choice == "efficientnet":
                    model = load_local_model(
                        "models/efficientnetb0_final.keras"
                    )
                else:
                    model = load_huggingface_model(model_choice)

                prediction, confidence, tumour_probability = (
                    predict_mri(image, model)
                )

            st.subheader("Prediction")

            if prediction == "Tumour":
                st.error(f"🧠 {prediction}")
            else:
                st.success(f"✅ {prediction}")

            st.metric(
                "Predicted Class Probability",
                f"{confidence:.2f}%"
            )

            st.caption(
                f"Tumour probability: {tumour_probability:.4f}"
            )

            st.info(
                "This is an experimental AI classification, "
                "not a medical diagnosis. A qualified radiologist "
                "must interpret MRI scans for clinical decisions."
            )

    except Exception as e:
        st.error(
            "The model could not process this image. "
            "Please check the model files and deployment logs."
        )
        st.exception(e)

import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import cv2
import numpy as np

# -----------------------------
# Skull Boundary Check
# -----------------------------
def is_valid_brain_mri(pil_image, min_area_ratio=0.05, max_area_ratio=0.85, min_circularity=0.3):
    """
    Performs a basic skull/brain boundary check to filter out non-MRI images.
    Returns True if the image looks like a valid axial brain MRI, False otherwise.
    """
    # Convert PIL image to grayscale numpy array
    img_array = np.array(pil_image.convert("L"))
    h, w = img_array.shape
    total_area = h * w
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
    
    # Otsu's thresholding to separate foreground (brain/skull) from background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
        
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    area_ratio = area / total_area
    
    # If the largest contour is basically the whole image, the colors might be inverted.
    # In that case, invert the threshold and try again.
    if area_ratio > 0.90:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        area_ratio = area / total_area

    # Check if area ratio is within expected bounds for a brain scan (not too small, not the whole image)
    if not (min_area_ratio < area_ratio < max_area_ratio):
        return False
        
    # Check circularity: 4 * pi * area / (perimeter^2). A perfect circle is 1.0.
    # Brain cross-sections are roughly elliptical, so a threshold of ~0.3-0.4 works well.
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter == 0:
        return False
        
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    if circularity < min_circularity:
        return False
        
    return True

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Load All Models for Consensus
# -----------------------------
REPO_ID = "abhishek2prince/brain-tumor-convnext"

@st.cache_resource
def load_all_models():
    models = {}
    
    # 1. EfficientNetB0 (Local)
    try:
        models["EfficientNetB0"] = tf.keras.models.load_model(
            "models/efficientnetb0_final.keras", compile=False
        )
    except Exception as e:
        st.error(f"Failed to load EfficientNetB0: {e}")

    # 2. ConvNeXt-Tiny (BCE Loss)
    try:
        models["ConvNeXt-BCE"] = load_huggingface_model("convnext_tiny_bce.keras")
    except Exception as e:
        st.error(f"Failed to load ConvNeXt-BCE: {e}")

    # 3. ConvNeXt-Tiny (Focal Loss)
    try:
        models["ConvNeXt-Focal"] = load_huggingface_model("convnext_tiny_focal.keras")
    except Exception as e:
        st.error(f"Failed to load ConvNeXt-Focal: {e}")

    return models

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

        # --- INVALID SCAN CHECK ---
        if not is_valid_brain_mri(image):
            st.error("⚠️ **Invalid Scan Detected**")
            st.warning("The uploaded image does not appear to be a valid axial brain MRI scan. Please upload a proper brain MRI image.")
            st.stop()  # Halts further execution
        # --------------------------------

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

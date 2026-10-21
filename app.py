import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import cv2

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Skull Boundary Check
# -----------------------------
def is_valid_brain_mri(pil_image, min_area_ratio=0.05, max_area_ratio=0.85, min_circularity=0.3):
    """
    Performs a basic skull/brain boundary check to filter out non-MRI images.
    Returns True if the image looks like a valid axial brain MRI, False otherwise.
    """
    img_array = np.array(pil_image.convert("L"))
    h, w = img_array.shape
    total_area = h * w
    
    blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    area_ratio = area / total_area
    
    if area_ratio > 0.90:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        area_ratio = area / total_area

    if not (min_area_ratio < area_ratio < max_area_ratio):
        return False
        
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter == 0:
        return False
        
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    if circularity < min_circularity:
        return False
        
    return True

# -----------------------------
# Model Loading Functions
# -----------------------------
REPO_ID = "abhishek2prince/brain-tumor-convnext"

@st.cache_resource
def load_huggingface_model(filename):
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model"
    )
    return tf.keras.models.load_model(model_path, compile=False)

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

# -----------------------------
# Prediction Function
# -----------------------------
def predict_mri(image, model):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    output = model.predict(image_array, verbose=0)
    tumour_probability = float(np.asarray(output).reshape(-1)[0])

    prediction = "Tumour" if tumour_probability >= 0.5 else "Non-Tumour"

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
    "Upload a brain MRI image or try a sample scan to obtain a binary tumour classification "
    "using our 3-model consensus engine."
)

# --- NEW: LAYOUT FOR UPLOAD & SAMPLE BUTTON ---
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )
with col2:
    # This button triggers the sample scan mode
    sample_mode = st.button(
        "🖼️ Try Sample Scan", 
        use_container_width=True,
        help="Click to test the app with a pre-loaded MRI scan."
    )
# ----------------------------------------------

# Determine which image to process
image_to_process = None
is_sample_scan = False

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert("RGB")
elif sample_mode:
    try:
        # Load the sample image from the assets folder
        image_to_process = Image.open("assets/sample_mri.jpg").convert("RGB")
        is_sample_scan = True
        st.info("🖼️ Running prediction on the **Sample Scan**.")
    except FileNotFoundError:
        st.error("❌ Sample image not found. Please ensure 'assets/sample_mri.jpg' exists.")

# --- MAIN PREDICTION FLOW ---
if image_to_process is not None:
    
    # 1. INVALID SCAN CHECK
    if not is_valid_brain_mri(image_to_process):
        st.error("⚠️ **Invalid Scan Detected**")
        st.warning("The uploaded image does not appear to be a valid axial brain MRI scan. Please upload a proper brain MRI image.")
        st.stop()

    # 2. DISPLAY IMAGE
    st.image(
        image_to_process,
        caption="Sample MRI" if is_sample_scan else "Uploaded MRI",
        use_container_width=True
    )

    # 3. CONSENSUS PREDICTION (From Step 2)
    if st.button("🔍 Run Consensus Prediction", type="primary"):
        with st.spinner("Running analysis across 3 models..."):
            models = load_all_models()
            
            if len(models) < 3:
                st.error("Not all models could be loaded. Please check your model files.")
                st.stop()

            results = []
            tumour_votes = 0

            for model_name, model in models.items():
                pred, conf, prob = predict_mri(image_to_process, model)
                results.append({
                    "model": model_name,
                    "prediction": pred,
                    "confidence": conf,
                    "probability": prob
                })
                if pred == "Tumour":
                    tumour_votes += 1

            non_tumour_votes = 3 - tumour_votes

        # --- DISPLAY RESULTS ---
        st.markdown("---")
        st.subheader("🧠 Final Consensus Prediction")

        if tumour_votes >= 2:
            consensus_class = "Tumourous"
            agreement = tumour_votes
        else:
            consensus_class = "Non-Tumourous"
            agreement = non_tumour_votes

        confidence_level = "High Confidence" if agreement == 3 else "Moderate Confidence"

        st.markdown(f"### {consensus_class} ({confidence_level})")
        st.info(f"**Consensus:** {agreement}/3 models agree this is a **{consensus_class.lower()}** scan ({confidence_level}).")

        st.subheader("📊 Individual Model Breakdown")
        for res in results:
            icon = "🧠" if res["prediction"] == "Tumour" else "✅"
            st.write(f"- **{res['model']}**: {icon} {res['prediction']} ({res['confidence']:.2f}% probability)")

        st.caption(
            "⚠️ This is an experimental AI classification, not a medical diagnosis. "
            "A qualified radiologist must interpret MRI scans for clinical decisions."
        )

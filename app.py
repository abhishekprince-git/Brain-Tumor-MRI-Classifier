import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError 
import cv2
from matplotlib import cm
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
# Skull Boundary Check
# -----------------------------
def is_valid_brain_mri(pil_image):
    """
    Robust check for valid brain MRI.
    Handles edge cases: white borders, inverted colors, solid shapes, and complex photos.
    """
    # Resize to a standard size for consistent math (faster and more stable)
    img_resized = pil_image.resize((512, 512))
    img_array = np.array(img_resized.convert("L"))
    h, w = img_array.shape
    total_area = h * w
    
    # EDGE CASE 1: Solid Color Check (e.g., a solid red circle)
    if np.std(img_array) < 15:
        return False

    # Thresholding
    blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    area_ratio = area / total_area
    
    # EDGE CASE 2: Inverted Colors / White Backgrounds
    # If the largest contour is basically the whole image, the colors might be inverted.
    if area_ratio > 0.90:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        area_ratio = area / total_area

    # Check if area ratio is within expected bounds for a brain scan
    if not (0.10 < area_ratio < 0.75):
        return False
        
    # Check circularity
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # A brain is an oval (0.4 to 0.95). Rejects perfect circles (1.0) and complex shapes like people (<0.4)
    if not (0.4 <= circularity <= 0.95):
        return False

    # EDGE CASE 3: Texture Check (Standard Deviation inside the brain)
    mask = np.zeros(img_array.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    mean, std_dev = cv2.meanStdDev(img_array, mask=mask)
    
    # Brain tissue has folds (high texture). A solid logo or smooth shape will have low std_dev.
    if std_dev[0] < 20:
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
    "Upload a brain MRI image or select a sample scan below. "
    "The 3-model consensus engine will predict the result instantly."
)

# --- 1. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

# --- 2. SESSION STATE ---
if 'selected_sample' not in st.session_state:
    st.session_state['selected_sample'] = None

# --- 3. DETERMINE IMAGE SOURCE ---
image_to_process = None
is_sample_scan = False

if uploaded_file is not None:
    image_to_process = Image.open(uploaded_file).convert("RGB")
    st.session_state['selected_sample'] = None 
    is_sample_scan = False
else:
    st.markdown("### Select a sample scan to test (Click an image):")
    col1, col2, col3, col4 = st.columns(4)
    
    samples = [
        ("assets/sample_1.jpg", "Sample 1"),
        ("assets/sample_2.jpg", "Sample 2"),
        ("assets/sample_3.jpg", "Sample 3"),
        ("assets/sample_4.jpg", "Sample 4")
    ]
    
    for col, (path, name) in zip([col1, col2, col3, col4], samples):
        with col:
            is_selected = (st.session_state['selected_sample'] == path)
            caption = "Selected" if is_selected else name
            st.image(path, caption=caption, use_container_width=True)
            
            btn_key = f"select_{path}"
            if st.button("Select this scan", key=btn_key, use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state['selected_sample'] = path
                st.rerun()

    if st.session_state['selected_sample']:
        try:
            image_to_process = Image.open(st.session_state['selected_sample']).convert("RGB")
            is_sample_scan = True
        except FileNotFoundError:
            st.error("Selected sample image not found.")
            st.stop()
    else:
        st.stop()

# --- 4. MAIN PREDICTION FLOW ---
if image_to_process is not None:
    
    if not is_valid_brain_mri(image_to_process):
        st.error("**Invalid Scan Detected**")
        st.warning("The image does not appear to be a valid axial brain MRI scan. Please upload a proper brain MRI image.")
        st.stop()

    st.markdown("---")
    
    with st.spinner(" Running analysis across 3 models..."):
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

    # --- DISPLAY CONSENSUS RESULTS ---
    if tumour_votes >= 2:
        consensus_class = "Tumourous"
        agreement = tumour_votes
    else:
        consensus_class = "Non-Tumourous"
        agreement = non_tumour_votes

    confidence_level = "High Confidence" if agreement == 3 else "Moderate Confidence"

    st.markdown(f"### {consensus_class} ({confidence_level})")
    st.info(f"**Consensus:** {agreement}/3 models agree this is a **{consensus_class.lower()}** scan ({confidence_level}).")

    st.subheader("Individual Model Breakdown")
    try:
        eff_model = models["EfficientNetB0"]
        last_conv_name = get_last_conv_layer_name(eff_model)
        
        if last_conv_name:
            # Prepare image array for the model (same as predict_mri)
            img_resized = image_to_process.resize((224, 224))
            img_array = np.expand_dims(np.asarray(img_resized, dtype=np.float32), axis=0)
            
            heatmap = get_gradcam_heatmap(img_array, eff_model, last_conv_name)
            superimposed_img = create_superimposed_image(image_to_process, heatmap)
            
            # Display side-by-side
            col_orig, col_heat = st.columns(2)
            with col_orig:
                st.image(image_to_process, caption="Original MRI", use_container_width=True)
            with col_heat:
                st.image(superimposed_img, caption="Grad-CAM Heatmap (EfficientNetB0)", use_container_width=True)
        else:
            st.warning("Could not find a convolutional layer for Grad-CAM.")
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")

st.caption(
    "⚠️ This is an experimental AI classification, not a medical diagnosis. "
    "A qualified radiologist must interpret MRI scans for clinical decisions."
)

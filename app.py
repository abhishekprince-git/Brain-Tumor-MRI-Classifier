import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
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
    img_resized = pil_image.resize((512, 512))
    img_array = np.array(img_resized.convert("L"))
    h, w = img_array.shape
    total_area = h * w
    
    if np.std(img_array) < 15:
        return False

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

    if not (0.10 < area_ratio < 0.75):
        return False
        
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    if not (0.4 <= circularity <= 0.95):
        return False

    mask = np.zeros(img_array.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    mean, std_dev = cv2.meanStdDev(img_array, mask=mask)
    
    if std_dev[0] < 20:
        return False

    return True

# -----------------------------
# Grad-CAM Functions
# -----------------------------
# Verified layer names from the GradCAM notebook (07_GradCAM.ipynb)
GRADCAM_LAYER_MAP = {
    "EfficientNetB0": "top_conv",
    "ConvNeXt-BCE": "convnext_tiny_stage_3_block_2_depthwise_conv",
    "ConvNeXt-Focal": "convnext_tiny_stage_3_block_2_depthwise_conv",
}

def get_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generates a Grad-CAM heatmap for the given image and model.
    Uses the tumour probability (single sigmoid output) as the target score.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)
        # For binary sigmoid output, target is the tumour probability (index 0)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize: ReLU then scale to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()

def create_superimposed_image(original_image, heatmap, alpha=0.4):
    """Overlays the heatmap onto the original image using JET colormap."""
    heatmap_resized = cv2.resize(heatmap, (original_image.width, original_image.height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    img_array = np.array(original_image.convert("RGB"))
    superimposed_img = heatmap_colored * alpha + img_array.astype(np.float32)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

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
    
    try:
        models["EfficientNetB0"] = tf.keras.models.load_model(
            "models/efficientnetb0_final.keras", compile=False
        )
    except Exception as e:
        st.error(f"Failed to load EfficientNetB0: {e}")

    try:
        models["ConvNeXt-BCE"] = load_huggingface_model("convnext_tiny_bce.keras")
    except Exception as e:
        st.error(f"Failed to load ConvNeXt-BCE: {e}")

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
st.title("Brain Tumor MRI Classifier")
st.write(
    "Upload a brain MRI image or select a sample scan below. "
    "The 3-model consensus engine will predict the result and generate an explainability heatmap."
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
    st.markdown("### Select a sample scan to test:")
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
            button_type = "primary" if is_selected else "secondary"
            if st.button("Select this scan", key=btn_key, use_container_width=True, type=button_type):
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
        st.error("Invalid Scan Detected")
        st.warning("The image does not appear to be a valid axial brain MRI scan. Please upload a proper brain MRI image.")
        st.stop()

    st.markdown("---")
    
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

    # --- DISPLAY CONSENSUS RESULTS ---
    if tumour_votes >= 2:
        consensus_class = "Tumourous"
        agreement = tumour_votes
    else:
        consensus_class = "Non-Tumourous"
        agreement = non_tumour_votes

    confidence_level = "High Confidence" if agreement == 3 else "Moderate Confidence"

    st.markdown(f"### {consensus_class} ({confidence_level})")
    st.info(f"**Consensus:** {agreement}/3 models agree this is a {consensus_class.lower()} scan ({confidence_level}).")

    st.subheader("Individual Model Breakdown")
    for res in results:
        st.write(f"- **{res['model']}**: {res['prediction']} ({res['confidence']:.2f}% probability)")
        
    # --- 5. GRAD-CAM HEATMAP (Using the model with highest confidence) ---
    st.markdown("---")
    st.subheader("Explainable AI: Tumor Localization (Grad-CAM)")
    st.write(
        "The heatmap below is generated by the model with the highest confidence score. "
        "It highlights the regions of the MRI that most strongly influenced the prediction."
    )
    
    # Pick the model with the highest confidence
    best_model_result = max(results, key=lambda r: r["confidence"])
    best_model_name = best_model_result["model"]
    best_model = models[best_model_name]
    
    st.write(f"**Using model:** {best_model_name} (confidence: {best_model_result['confidence']:.2f}%)")
    
    try:
        # Get the verified Grad-CAM layer name for this model
        gradcam_layer_name = GRADCAM_LAYER_MAP.get(best_model_name)
        
        if gradcam_layer_name is None:
            st.warning(f"No Grad-CAM layer mapping defined for {best_model_name}.")
        else:
            # Verify the layer exists in the loaded model
            try:
                layer = best_model.get_layer(gradcam_layer_name)
            except ValueError:
                st.warning(f"Layer '{gradcam_layer_name}' not found in {best_model_name}. Available layers: {[l.name for l in best_model.layers[:5]]}...")
                layer = None
            
            if layer is not None:
                img_resized = image_to_process.resize((224, 224))
                img_array = np.expand_dims(np.asarray(img_resized, dtype=np.float32), axis=0)
                
                heatmap = get_gradcam_heatmap(img_array, best_model, gradcam_layer_name)
                superimposed_img = create_superimposed_image(image_to_process, heatmap)
                
                col_orig, col_heat = st.columns(2)
                with col_orig:
                    st.image(image_to_process, caption="Original MRI", use_container_width=True)
                with col_heat:
                    st.image(superimposed_img, caption=f"Grad-CAM Heatmap ({best_model_name})", use_container_width=True)
                
                # Show the raw heatmap separately for clarity
                st.markdown("**Raw attention map:**")
                heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap, (image_to_process.width, image_to_process.height)))
                st.image(heatmap_uint8, caption="Normalized Grad-CAM Heatmap", use_container_width=True)
            else:
                st.warning(f"Could not locate the Grad-CAM target layer in {best_model_name}.")
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")

st.markdown("---")
st.caption(
    "Disclaimer: This is an experimental AI classification tool and not a medical diagnosis. "
    "A qualified radiologist must interpret MRI scans for all clinical decisions."
)

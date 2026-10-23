import os
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

IMG_SIZE = (224, 224)
THRESHOLD = 0.5
MODEL_PATH = os.environ.get(
    "BRAIN_TUMOR_MODEL_PATH",
    str(Path(__file__).parent / "models" / "efficientnetb0_final.keras"),
)

# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource(show_spinner="Loading EfficientNetB0 model...")
def load_model(model_path: str):
    """Load the saved Keras model once per Streamlit session."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Place your trained model at "
            "'models/efficientnetb0_final.keras' or set BRAIN_TUMOR_MODEL_PATH."
        )
    return tf.keras.models.load_model(str(path), compile=False)

# -----------------------------
# Image preprocessing and prediction
# -----------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prepare RGB input. This assumes training used float32 RGB pixels in [0, 255]."""
    resized = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_binary(model, batch: np.ndarray):
    """Return tumour probability, predicted label, and predicted-class score."""
    output = model(batch, training=False)
    output = np.asarray(output).reshape(-1)

    if output.size != 1:
        raise ValueError(
            f"This app expects a binary model with one output, but got "
            f"{output.size} outputs. Check model.output_shape."
        )

    tumour_probability = float(np.clip(output[0], 0.0, 1.0))
    is_tumour = tumour_probability >= THRESHOLD
    predicted_label = "Tumourous" if is_tumour else "Non-Tumourous"
    predicted_score = tumour_probability if is_tumour else 1.0 - tumour_probability
    return tumour_probability, predicted_label, predicted_score

# -----------------------------
# Grad-CAM helpers
# -----------------------------
def find_last_conv_layer(model):
    """Find the last Conv2D/DepthwiseConv2D layer and its top-level nested owner."""
    candidates = []

    def walk(layer, owner=None):
        for child in getattr(layer, "layers", []):
            if isinstance(child, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                candidates.append((child, owner))
            if getattr(child, "layers", None):
                # The first nested model below the outer model owns this layer.
                next_owner = child if owner is None else owner
                walk(child, next_owner)

    # Include top-level convolutional layers and nested backbones in layer order.
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            candidates.append((layer, None))
        if getattr(layer, "layers", None):
            walk(layer, layer)

    if not candidates:
        return None, None
    return candidates[-1]

def build_gradcam_model(model, target_layer, backbone):
    """
    Return a model producing [target activations, final predictions].
    For nested backbones, build the activation graph within the backbone and
    replay the outer classifier layers after that backbone.
    """
    if backbone is None:
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[target_layer.output, backbone.output],
    )

    model_input = model.inputs[0]
    activations, features = backbone_grad_model(model_input, training=False)
    x = features

    try:
        backbone_index = model.layers.index(backbone)
    except ValueError as exc:
        raise ValueError(
            "The selected convolutional layer's backbone is not a direct "
            "child of the outer model. An architecture-specific Grad-CAM "
            "path is needed for this saved model."
        ) from exc

    for layer in model.layers[backbone_index + 1:]:
        x = layer(x)

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[activations, x],
    )
    
def make_gradcam_heatmap(batch: np.ndarray, model, target_layer, backbone, tumour_class: bool):
    """Create a normalized Grad-CAM heatmap for the selected binary class."""
    grad_model = build_gradcam_model(model, target_layer, backbone)
    image_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_activations, predictions = grad_model(image_tensor, training=False)
        tumour_score = predictions[:, 0]
        class_score = tumour_score if tumour_class else (1.0 - tumour_score)

    gradients = tape.gradient(class_score, conv_activations)
    if gradients is None:
        raise ValueError(
            "Gradients are disconnected from the selected convolutional layer. "
            "Try a different target layer or inspect the saved model architecture."
        )

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * conv_activations, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0.0)
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(
        max_value > 0,
        heatmap / (max_value + tf.keras.backend.epsilon()),
        tf.zeros_like(heatmap),
    )
    return heatmap.numpy()

def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.38):
    """Overlay a Grad-CAM heatmap on the original MRI image."""
    original = np.asarray(image.convert("RGB"))
    resized_heatmap = cv2.resize(
        heatmap, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    heatmap_uint8 = np.uint8(np.clip(resized_heatmap, 0, 1) * 255)
    # OpenCV's JET colormap is BGR; convert to RGB before overlay.
    colored = cv2.cvtColor(
        cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    overlay = (1.0 - alpha) * original.astype(np.float32) + alpha * colored.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8), heatmap_uint8

# -----------------------------
# User interface
# -----------------------------
st.title("Brain MRI Classification")
st.write(
    "Upload a brain MRI image to obtain a binary model prediction and a Grad-CAM "
    "visualization of image regions that influenced that prediction."
)

st.info(
    "Research demonstration only. This app is not a medical diagnostic tool. "
    "A qualified radiologist must interpret MRI scans and make clinical decisions."
)

with st.sidebar:
    st.header("Model settings")
    st.write("Model: EfficientNetB0")
    st.write(f"Decision threshold: {THRESHOLD:.2f}")
    st.caption("The model must be the same binary model used during training.")
    st.caption("Preprocessing assumes RGB, 224 × 224, float32 pixel values in [0, 255].")

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["png", "jpg", "jpeg"],
    help="Use an MRI image compatible with the data used to train the model.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        st.stop()

    st.subheader("Uploaded image")
    st.image(image, caption="Input MRI", width=420)

    if st.button("Analyze MRI", type="primary"):
        try:
            model = load_model(MODEL_PATH)
            batch = preprocess_image(image)

            with st.spinner("Running prediction..."):
                tumour_probability, predicted_label, predicted_score = predict_binary(model, batch)

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted class", predicted_label)
            col2.metric("Predicted-class score", f"{predicted_score * 100:.2f}%")
            col3.metric("Tumour probability", f"{tumour_probability * 100:.2f}%")

            st.subheader("Binary class probabilities")
            st.progress(float(tumour_probability), text=f"Tumourous: {tumour_probability * 100:.2f}%")
            st.progress(
                float(1.0 - tumour_probability),
                text=f"Non-Tumourous: {(1.0 - tumour_probability) * 100:.2f}%",
            )

            st.caption(
                "These are model outputs, not necessarily calibrated clinical probabilities. "
                "A binary model cannot provide separate glioma, meningioma, and pituitary probabilities."
            )

            st.divider()
            st.subheader("Explainable AI: Grad-CAM")
            st.write(
                "Grad-CAM highlights image regions that influenced the selected class score. "
                "It is not a tumour segmentation mask."
            )

            try:
                target_layer, backbone = find_last_conv_layer(model)
                if target_layer is None:
                    st.warning("No Conv2D or DepthwiseConv2D layer was found in this model.")
                else:
                    st.caption(f"Target layer: {target_layer.name}")
                    with st.spinner("Generating Grad-CAM heatmap..."):
                        heatmap = make_gradcam_heatmap(
                            batch,
                            model,
                            target_layer,
                            backbone,
                            tumour_class=(predicted_label == "Tumourous"),
                        )
                        overlay, raw_heatmap = overlay_heatmap(image, heatmap)

                    original_col, overlay_col, raw_col = st.columns(3)
                    with original_col:
                        st.image(image, caption="Original MRI", use_container_width=True)
                    with overlay_col:
                        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
                    with raw_col:
                        st.image(raw_heatmap, caption="Grad-CAM heatmap", use_container_width=True)

            except Exception as exc:
                st.warning(f"Prediction succeeded, but Grad-CAM failed: {exc}")
                st.code(
                    "If this continues, share the full terminal traceback and "
                    "the model.summary() output so the target layer can be selected "
                    "for this exact architecture."
                )

        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not analyze this image: {exc}")
            st.caption("Check the model path, dependencies, and preprocessing used during training.")
st.divider()
st.caption(
    "Educational research application • Not validated for clinical diagnosis • "
    "Grad-CAM provides qualitative model explanations only"
)

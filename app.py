import hashlib
import io
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================================================
# Configuration
# =====================================================
st.set_page_config(
    page_title="Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = (224, 224)
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
ASSETS_DIR = PROJECT_DIR / "assets"

MODEL_NAME = "EfficientNetB0"
MODEL_PATH = MODEL_DIR / "efficientnetb0_final.keras"
SAMPLE_IMAGES = [ASSETS_DIR / f"sample_{i}.jpg" for i in range(1, 5)]

# Optional: assets/metrics.json, e.g.
# {"accuracy": 0.9573, "precision": 0.99, "recall": 0.95, "f1": 0.97}
METRICS_PATH = ASSETS_DIR / "metrics.json"

GREEN, RED, AMBER = "#2e9e6b", "#d9534f", "#e0a030"

# Newer Streamlit replaced use_container_width with width="stretch"
_v = tuple(int(p) for p in st.__version__.split(".")[:2] if p.isdigit())
STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}


# =====================================================
# Model loading
# =====================================================
@st.cache_resource(show_spinner=f"Loading {MODEL_NAME} model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


# =====================================================
# Brain MRI Validation (OpenCV)
# =====================================================
def is_valid_brain_mri(image: Image.Image):
    """
    Heuristic validation to help reject invalid or non-brain images.
    Checks image standard deviation, largest contour, circularity,
    and contour area ratio.

    Returns:
        (is_valid, message)
    """
    try:
        # Convert image to grayscale
        gray = np.array(image.convert("L"))

        # Reject images with insufficient intensity variation
        if gray.std() < 10:
            return False, "Image has insufficient intensity variation."

        # Apply Gaussian blur and Otsu thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find external contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False, "No significant image contour detected."

        # Select the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        contour_area = cv2.contourArea(largest_contour)
        image_area = gray.shape[0] * gray.shape[1]

        if image_area == 0 or contour_area <= 0:
            return False, "Invalid image dimensions or contour."

        # Calculate contour area ratio
        area_ratio = contour_area / image_area

        # Calculate circularity
        perimeter = cv2.arcLength(largest_contour, True)

        if perimeter <= 0:
            return False, "Invalid contour perimeter."

        circularity = (
            4 * np.pi * contour_area / (perimeter ** 2)
        )

        # Heuristic checks; calibrate against your MRI dataset
        if area_ratio < 0.10 or area_ratio > 0.95:
            return False, "Image contour area is outside the expected range."

        if circularity < 0.10:
            return False, "Image contour shape is unsuitable."

        return True, "Image passed basic structural validation."

    except Exception as exc:
        return False, f"Image validation failed: {exc}"


# =====================================================
# Preprocessing and prediction
# =====================================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    """RGB, 224x224, float32 pixels in [0, 255] (EfficientNet rescales internally)."""
    resized = image.convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)


def predict_tumour_probability(model, batch) -> float:
    output = np.asarray(model(batch, training=False)).reshape(-1)
    if output.size != 1:
        raise ValueError(
            f"Expected one sigmoid output for binary classification; got {output.size}."
        )
    return float(np.clip(output[0], 0.0, 1.0))


def build_result(tumour_probability: float, threshold: float) -> dict:
    is_tumour = tumour_probability >= threshold
    return {
        "is_tumour": is_tumour,
        "label": "Tumourous" if is_tumour else "Non-Tumourous",
        "tumour_probability": tumour_probability,
        "non_tumour_probability": 1.0 - tumour_probability,
        "confidence": tumour_probability if is_tumour else 1.0 - tumour_probability,
    }


# =====================================================
# Grad-CAM helpers
# =====================================================
def find_last_conv_layer(model):
    """Last Conv2D/DepthwiseConv2D layer and the nested backbone that owns it."""
    candidates = []
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)

    def walk(layer, owner=None):
        for child in getattr(layer, "layers", []):
            if isinstance(child, conv_types):
                candidates.append((child, owner))
            if getattr(child, "layers", None):
                walk(child, child if owner is None else owner)

    for layer in model.layers:
        if isinstance(layer, conv_types):
            candidates.append((layer, None))
        if getattr(layer, "layers", None):
            walk(layer, layer)

    if not candidates:
        return None, None
    return candidates[-1]


def build_gradcam_model(model, target_layer, backbone):
    """Model returning [target activations, final predictions]."""
    if backbone is None:
        return tf.keras.Model(
            inputs=model.inputs, outputs=[target_layer.output, model.output]
        )

    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs, outputs=[target_layer.output, backbone.output]
    )
    activations, features = backbone_grad_model(model.inputs[0], training=False)
    x = features

    try:
        backbone_index = model.layers.index(backbone)
    except ValueError as exc:
        raise ValueError(
            "The convolutional layer's backbone is not a direct child of the "
            "outer model; an architecture-specific Grad-CAM path is needed."
        ) from exc

    for layer in model.layers[backbone_index + 1:]:
        x = layer(x)

    return tf.keras.Model(inputs=model.inputs, outputs=[activations, x])


def make_gradcam_heatmap(batch, grad_model, tumour_class: bool) -> np.ndarray:
    """Normalized Grad-CAM heatmap (0-1) for the tumour or the non-tumour class."""
    image_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_activations, predictions = grad_model(image_tensor, training=False)
        tumour_score = predictions[:, 0]
        class_score = tumour_score if tumour_class else (1.0 - tumour_score)

    gradients = tape.gradient(class_score, conv_activations)
    if gradients is None:
        raise ValueError("Gradients are disconnected from the selected layer.")

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


def colorize_heatmap(heatmap: np.ndarray, size) -> np.ndarray:
    resized = cv2.resize(heatmap, size, interpolation=cv2.INTER_LINEAR)
    uint8 = np.uint8(np.clip(resized, 0, 1) * 255)
    return cv2.cvtColor(cv2.applyColorMap(uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float):
    original = np.asarray(image.convert("RGB"))
    colored = colorize_heatmap(heatmap, (original.shape[1], original.shape[0]))
    blended = (1.0 - alpha) * original.astype(np.float32) + alpha * colored.astype(np.float32)
    return colored, np.clip(blended, 0, 255).astype(np.uint8)


def to_png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# =====================================================
# Analysis
# =====================================================
def run_analysis(model, image: Image.Image) -> dict:
    start = time.time()
    batch = preprocess_image(image)
    tumour_probability = predict_tumour_probability(model, batch)

    analysis = {
        "tumour_probability": tumour_probability,
        "heat_tumour": None,
        "heat_clear": None,
        "layer_name": None,
        "gradcam_error": None,
    }
    try:
        target_layer, backbone = find_last_conv_layer(model)
        if target_layer is None:
            raise ValueError("No convolutional target layer was found.")
        grad_model = build_gradcam_model(model, target_layer, backbone)
        analysis["heat_tumour"] = make_gradcam_heatmap(batch, grad_model, True)
        analysis["heat_clear"] = make_gradcam_heatmap(batch, grad_model, False)
        analysis["layer_name"] = target_layer.name
    except Exception as exc:  # Grad-CAM must never break the prediction
        analysis["gradcam_error"] = str(exc)

    analysis["seconds"] = time.time() - start
    return analysis


def image_key(image: Image.Image) -> str:
    return hashlib.md5(image.convert("RGB").tobytes()).hexdigest()


# =====================================================
# UI helpers
# =====================================================
def probability_bar(label: str, fraction: float, color: str, mark: float = None) -> str:
    pct = fraction * 100
    marker = f'<div class="bar-mark" style="left:{mark * 100:.1f}%"></div>' if mark is not None else ""
    return f"""
    <div class="bar-wrap">
      <div class="bar-head"><span>{label}</span><span>{pct:.2f}%</span></div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{pct:.2f}%; background:{color};"></div>{marker}
      </div>
    </div>
    """


def confidence_level(confidence: float):
    if confidence >= 0.90:
        return "High", GREEN
    if confidence >= 0.70:
        return "Moderate", AMBER
    return "Low", RED


def load_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return None


def stat_cards(metrics: dict) -> str:
    cards = "".join(
        f'<div class="stat-card"><div class="v">{float(v) * 100:.1f}%</div>'
        f'<div class="l">{k.replace("_", " ")}</div></div>'
        for k, v in metrics.items()
    )
    return (
        f'<div style="display:grid;grid-template-columns:repeat({len(metrics)},1fr);'
        f'gap:.8rem;margin-bottom:1rem">{cards}</div>'
    )


def build_report(source_name, res, analysis, threshold) -> bytes:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image": source_name,
        "model": MODEL_NAME,
        "threshold": threshold,
        "predicted_class": res["label"],
        "tumourous_percent": round(res["tumour_probability"] * 100, 2),
        "non_tumourous_percent": round(res["non_tumour_probability"] * 100, 2),
        "gradcam_layer": analysis["layer_name"],
        "disclaimer": "Research demonstration only. Not a medical device.",
    }
    return json.dumps(report, indent=2).encode("utf-8")


# =====================================================
# Hero + sidebar
# =====================================================
metrics = load_metrics()

st.markdown(
    f"""
    <div class="hero">
      <h1>🧠 Brain MRI Classifier</h1>
      <p>Upload a brain MRI slice to get a Tumourous / Non-Tumourous prediction,
         class probabilities, and a Grad-CAM view of the regions that influenced the model.</p>
      <span class="pill">Deep learning</span>
      <span class="pill">{MODEL_NAME}</span>
      <span class="pill">Transfer learning</span>
      <span class="pill">Explainable AI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if metrics:
    st.markdown(stat_cards(metrics), unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Decision threshold", 0.10, 0.90, 0.50, 0.05,
        help="Images with a tumour probability at or above this value are labelled "
        "Tumourous. Lowering it catches more tumours (higher recall) but gives more "
        "false alarms.",
    )
    overlay_alpha = st.slider(
        "Grad-CAM overlay strength", 0.10, 0.80, 0.40, 0.05,
        help="How strongly the heatmap is blended over the MRI.",
    )
    show_both = st.toggle(
        "Show Grad-CAM for both classes", value=False,
        help="Compare what pushes the model towards Tumourous vs Non-Tumourous.",
    )
    st.divider()
    st.caption(f"Model: **{MODEL_NAME}**")
    st.caption("Input: RGB, 224 × 224, pixel values 0–255")
    st.caption("Task: binary classification (Tumourous vs Non-Tumourous)")
    if st.session_state.get("history"):
        if st.button("🗑️ Clear history", **STRETCH):
            st.session_state["history"] = []
            st.rerun()

st.warning(
    "⚠️ **Research demonstration only.** This is not a medical device and must not be "
    "used for diagnosis. MRI scans must be interpreted by a qualified radiologist.",
)

tab_analyze, tab_history, tab_how, tab_about = st.tabs(
    ["🔍 Analyze", "🕘 History", "🧪 How it works", "ℹ️ About & limitations"]
)

# =====================================================
# Tab 1: Analyze
# =====================================================
with tab_analyze:
    left, right = st.columns([1, 1.15], gap="large")

    image, source_name = None, None
    with left:
        st.subheader("Choose an MRI")
        available_samples = [p for p in SAMPLE_IMAGES if p.exists()]
        options = ["Upload my own image"] + [p.name for p in available_samples]
        source = st.selectbox("Image source", options)

        if source == "Upload my own image":
            uploaded = st.file_uploader(
                "Upload a brain MRI (PNG / JPG)", type=["png", "jpg", "jpeg"]
            )
            if uploaded is not None:
                try:
                    image = Image.open(uploaded).convert("RGB")
                    source_name = uploaded.name
                except Exception:
                    st.error("That file could not be read as an image.")
        else:
            chosen = next(p for p in available_samples if p.name == source)
            image = Image.open(chosen).convert("RGB")
            source_name = chosen.name

        if image is not None:
            st.image(image, caption=f"Selected MRI · {image.width}×{image.height}px", **STRETCH)
            analyze_clicked = st.button("Analyze MRI", type="primary", **STRETCH)
        else:
            analyze_clicked = False
            st.info("Select a sample or upload an MRI image to begin.")

    if image is not None and analyze_clicked:
        # Validate the uploaded or selected image before classification
        is_valid, validation_message = is_valid_brain_mri(image)

        if not is_valid:
            st.error(
                "Invalid or potentially non-brain image. "
                "Please upload a suitable brain MRI."
            )
            st.caption(f"Validation details: {validation_message}")
            st.stop()

        try:
            model = load_model()
        except Exception as exc:
            st.error(f"Could not load the model: {exc}")
            st.stop()

        with st.spinner("Running prediction and generating Grad-CAM..."):
            try:
                result = run_analysis(model, image)
                result["key"] = image_key(image)
                st.session_state["analysis"] = result

                st.session_state.setdefault("history", []).append(
                    {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Image": source_name,
                        "Tumourous %": round(
                            result["tumour_probability"] * 100, 2
                        ),
                        "Non-Tumourous %": round(
                            (1 - result["tumour_probability"]) * 100, 2
                        ),
                        "Threshold": threshold,
                    }
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

    analysis = st.session_state.get("analysis")
    has_result = (
        image is not None and analysis is not None and analysis["key"] == image_key(image)
    )

    with right:
        st.subheader("Result")
        if not has_result:
            st.caption("The prediction will appear here after you click **Analyze MRI**.")
        else:
            res = build_result(analysis["tumour_probability"], threshold)
            card_class = "tumour" if res["is_tumour"] else "clear"
            level, level_color = confidence_level(res["confidence"])

            st.markdown(
                f"""
                <div class="result-card {card_class}">
                  <div class="result-label">Predicted class</div>
                  <div class="result-value">{res['label']}</div>
                  <div class="result-sub">
                    Predicted-class score: <b>{res['confidence'] * 100:.2f}%</b> ·
                    Confidence: <b style="color:{level_color}">{level}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="prob-line">
                  <div class="prob-chip t">Tumourous: {res['tumour_probability'] * 100:.2f}%</div>
                  <div class="prob-chip n">Non-Tumourous: {res['non_tumour_probability'] * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                probability_bar("Tumourous", res["tumour_probability"], RED, mark=threshold)
                + probability_bar("Non-Tumourous", res["non_tumour_probability"], GREEN),
                unsafe_allow_html=True,
            )
            st.caption("The vertical tick on the Tumourous bar marks the decision threshold.")

            if 0.35 <= res["tumour_probability"] <= 0.65:
                st.warning(
                    "The model is close to the decision boundary for this image. "
                    "Treat this prediction with extra caution."
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Decision threshold", f"{threshold:.2f}")
            c2.metric("Confidence", level)
            c3.metric("Inference time", f"{analysis['seconds']:.1f} s")

            st.download_button(
                "⬇️ Download report (JSON)",
                data=build_report(source_name, res, analysis, threshold),
                file_name="mri_prediction_report.json",
                mime="application/json",
                **STRETCH,
            )
            st.caption(
                "Scores are model outputs, not calibrated clinical probabilities, "
                "and do not prove the prediction is correct."
            )

    # ---- Grad-CAM (full width) ----
    if has_result:
        st.divider()
        st.subheader("3 · Explainable AI: Grad-CAM")

        if analysis["gradcam_error"]:
            st.warning(f"Grad-CAM could not be generated: {analysis['gradcam_error']}")
        else:
            res = build_result(analysis["tumour_probability"], threshold)
            heatmap = analysis["heat_tumour"] if res["is_tumour"] else analysis["heat_clear"]
            colored, overlay = overlay_heatmap(image, heatmap, overlay_alpha)

            st.caption(
                f"Heatmap for the predicted class (**{res['label']}**) · "
                f"Model: {MODEL_NAME} · Target layer: `{analysis['layer_name']}`"
            )

            g1, g2, g3 = st.columns(3)
            g1.image(image, caption="Original MRI", **STRETCH)
            g2.image(colored, caption="Grad-CAM heatmap", **STRETCH)
            g3.image(overlay, caption="Overlay", **STRETCH)

            if show_both:
                st.markdown("**Class comparison**")
                _, ov_t = overlay_heatmap(image, analysis["heat_tumour"], overlay_alpha)
                _, ov_n = overlay_heatmap(image, analysis["heat_clear"], overlay_alpha)
                b1, b2 = st.columns(2)
                b1.image(ov_t, caption="Evidence for Tumourous", **STRETCH)
                b2.image(ov_n, caption="Evidence for Non-Tumourous", **STRETCH)

            st.caption(
                "Warmer colours (red/yellow) mark regions that pushed the model towards "
                "its prediction. This is **not** a tumour segmentation mask, and the "
                "highlighted area is not guaranteed to be the true lesion."
            )
            st.download_button(
                "⬇️ Download overlay (PNG)",
                data=to_png_bytes(overlay),
                file_name="gradcam_overlay.png",
                mime="image/png",
            )

# =====================================================
# Tab 2: History
# =====================================================
with tab_history:
    st.subheader("Session history")
    history = st.session_state.get("history", [])
    if not history:
        st.caption("Analyses you run in this session will be listed here.")
    else:
        df = pd.DataFrame(history)
        st.dataframe(df, hide_index=True, **STRETCH)
        st.download_button(
            "⬇️ Download history (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="analysis_history.csv",
            mime="text/csv",
        )
        st.line_chart(df.set_index(df.index + 1)[["Tumourous %"]])

# =====================================================
# Tab 3: How it works
# =====================================================
with tab_how:
    st.subheader("From MRI slice to prediction")
    steps = [
        ("1 · Preprocess", "The image is converted to RGB and resized to 224 × 224 pixels."),
        ("2 · Extract features", f"{MODEL_NAME}, pre-trained on ImageNet, extracts visual features from the MRI."),
        ("3 · Classify", "A sigmoid output gives the probability that the scan is Tumourous."),
        ("4 · Explain", "Grad-CAM highlights image regions that influenced the prediction."),
    ]
    for col, (title, text) in zip(st.columns(4), steps):
        with col:
            st.markdown(
                f'<div class="step-card"><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown(
        """
**Decision threshold.** The model outputs a probability. By default, anything at or
above 0.50 is labelled *Tumourous*. In a screening-style problem, missing a tumour is
usually worse than a false alarm, so a lower threshold (the sidebar slider) trades some
precision for higher recall.

**Grad-CAM.** Gradients of the predicted class are propagated back to the last
convolutional layer. Averaging them gives importance weights for each feature map, and
the weighted sum is shown as a heatmap over the original image.
        """
    )

    if metrics:
        st.subheader("Test-set performance")
        st.markdown(stat_cards(metrics), unsafe_allow_html=True)

    with st.expander("FAQ"):
        st.markdown(
            """
- **Why two percentages?** They are the model's output for each class and always add up to 100%.
- **Does a high score mean the result is right?** No. Scores are not calibrated and can be confidently wrong.
- **What should I upload?** A single 2D brain MRI slice as PNG or JPG.
- **Is my image stored?** It is only processed in this session's memory.
            """
        )

# =====================================================
# Tab 4: About
# =====================================================
with tab_about:
    st.subheader("About this project")
    st.markdown(
        f"""
This app classifies brain MRI slices as **Tumourous** or **Non-Tumourous** using
transfer learning ({MODEL_NAME}). It was trained on a public brain-tumour MRI dataset
in which glioma, meningioma and pituitary tumour images form the *Tumourous* class.

**Limitations**
- Binary only: it does not identify the tumour type or location.
- Trained on a single public dataset; performance on scans from other scanners,
  hospitals or imaging protocols is unknown.
- The dataset contains augmented images, so reported scores may be optimistic.
- Grad-CAM shows model attention, not a verified lesion boundary.
- It should not be used for diagnosis, screening or any clinical decision.
        """
    )
    st.link_button(
        "View source on GitHub",
        "https://github.com/abhishekprince-git/Brain-Tumor-MRI-Classifier",
    )

st.markdown(
    '<div class="footer">Built with Streamlit & TensorFlow · For research and education only</div>',
    unsafe_allow_html=True,
)

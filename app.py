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
    # Simplified and robust TF2 approach: directly reference the target

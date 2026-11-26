import hashlib
import html
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# =====================================================
# Configuration
# =====================================================
st.set_page_config(
    page_title="Brain MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
ASSETS_DIR = PROJECT_DIR / "assets"
RESULTS_DIR = PROJECT_DIR / "results"

# --- Model selection -------------------------------------------------------
# The app picks the first match, in this order. ResNet files come first because
# the latest notebook (ResNet + Grad-CAM) is the model that should be served.
# Inside one pattern, files with "final"/"best" in the name win.
MODEL_PATTERNS = ["*resnet*.keras", "*efficientnet*.keras", "*.keras", "*.h5"]
MODEL_FOLDERS = [MODEL_DIR, PROJECT_DIR, RESULTS_DIR]

# How pixels are fed to the network. None = auto-detect (see detect_preprocess_mode).
#   "raw"     -> pixels 0-255 unchanged (EfficientNet, or models with built-in rescaling)
#   "caffe"   -> Keras ResNet/VGG preprocess_input (RGB->BGR, subtract ImageNet mean)
#   "rescale" -> divide by 255
# Set this to match what the notebook did at training time.
PREPROCESS_OVERRIDE = None

# If the model has 2 softmax outputs, this is the index of the "Tumourous" class.
TUMOUR_INDEX = 1

# --- Upload limits -----------------------------------------------------------
ALLOWED_FORMATS = {"JPEG", "PNG", "MPO", "WEBP", "BMP"}
MAX_UPLOAD_MB = 15
MAX_PIXELS = 40_000_000       # protects against decompression bombs
MIN_SIDE = 64                 # px; smaller images are rejected
DISPLAY_MAX_SIDE = 1024       # large images are downscaled once on load

SAMPLE_IMAGES = [ASSETS_DIR / f"sample_{i}.jpg" for i in range(1, 5)]
UPLOAD_OPTION = "Upload my own image"

# Optional: put test-set results in assets/metrics.json (or results/metrics.json), e.g.
# {"accuracy": 0.9573, "precision": 0.99, "recall": 0.95, "f1": 0.97}
# If no file is found, the performance section is simply hidden.
METRICS_CANDIDATES = [ASSETS_DIR / "metrics.json", RESULTS_DIR / "metrics.json"]

GREEN = "#2e9e6b"
RED = "#d9534f"
AMBER = "#e0a030"

# =====================================================
# Styling (works in light and dark themes)
# =====================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px;}

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(88,101,242,0.22), rgba(46,158,107,0.18));
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1.4rem;
    }
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2.2rem;}
    .hero p {margin: 0; opacity: .85; font-size: 1.02rem;}

    .pill {
        display: inline-block; padding: .18rem .7rem; margin: .8rem .4rem 0 0;
        border-radius: 999px; font-size: .78rem; font-weight: 600;
        background: rgba(128,128,128,0.18); border: 1px solid rgba(128,128,128,0.25);
    }

    .result-card {
        padding: 1.3rem 1.5rem; border-radius: 16px; margin-bottom: 1rem;
        border: 1px solid rgba(128,128,128,0.25);
    }
    .result-card.tumour {background: rgba(217,83,79,0.13); border-color: rgba(217,83,79,0.5);}
    .result-card.clear  {background: rgba(46,158,107,0.13); border-color: rgba(46,158,107,0.5);}
    .result-card.invalid {background: rgba(224,160,48,0.14); border-color: rgba(224,160,48,0.6);}
    .result-card ul {margin: .5rem 0 0 1.1rem; padding: 0;}
    .result-label {font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; opacity: .7;}
    .result-value {font-size: 2rem; font-weight: 700; line-height: 1.2; margin: .15rem 0 .3rem 0;}
    .result-sub {opacity: .8; font-size: .95rem;}

    .bar-wrap {margin: .55rem 0;}
    .bar-head {display: flex; justify-content: space-between; font-weight: 600; margin-bottom: .25rem;}
    .bar-track {width: 100%; height: 14px; border-radius: 999px; background: rgba(128,128,128,0.2); overflow: hidden;}
    .bar-fill {height: 100%; border-radius: 999px;}

    .step-card {
        padding: 1.1rem 1.2rem; border-radius: 14px; height: 100%;
        background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.2);
    }
    .step-card h4 {margin: 0 0 .4rem 0;}
    .footer {text-align: center; opacity: .6; font-size: .85rem; margin-top: 2.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Streamlit version-compat helpers
# (use_container_width was replaced by width="stretch" in newer Streamlit;
#  older versions raise TypeError *or* StreamlitAPIException, so catch both)
# =====================================================
def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except Exception:
        st.image(img, caption=caption, use_container_width=True)


def primary_button(label: str, disabled: bool = False) -> bool:
    try:
        return st.button(label, type="primary", width="stretch", disabled=disabled)
    except Exception:
        return st.button(label, type="primary", use_container_width=True, disabled=disabled)


# =====================================================
# Model discovery, loading and preprocessing mode
# =====================================================
def resolve_model_path() -> Path:
    for pattern in MODEL_PATTERNS:
        for folder in MODEL_FOLDERS:
            if not folder.exists():
                continue
            found = sorted(
                folder.glob(pattern),
                key=lambda p: (
                    not any(tag in p.name.lower() for tag in ("final", "best")),
                    p.name.lower(),
                ),
            )
            if found:
                return found[0]
    searched = ", ".join(str(f) for f in MODEL_FOLDERS)
    raise FileNotFoundError(
        f"No model file (.keras / .h5) found. Looked in: {searched}"
    )


def model_display_name(path: Path) -> str:
    stem = path.stem.lower()
    match = re.search(r"resnet\s*_?(\d+)", stem)
    if match:
        return f"ResNet{match.group(1)}"
    if "resnet" in stem:
        return "ResNet50"
    match = re.search(r"efficientnet\s*_?(b\d)", stem)
    if match:
        return f"EfficientNet{match.group(1).upper()}"
    if "efficientnet" in stem:
        return "EfficientNetB0"
    return path.stem


try:
    MODEL_PATH_PREVIEW = resolve_model_path()
    MODEL_NAME = model_display_name(MODEL_PATH_PREVIEW)
except FileNotFoundError:
    MODEL_PATH_PREVIEW = None
    MODEL_NAME = "CNN"


def _walk_layers(model):
    for layer in getattr(model, "layers", []):
        yield layer
        if getattr(layer, "layers", None):
            yield from _walk_layers(layer)


def detect_preprocess_mode(model, path: Path) -> str:
    if PREPROCESS_OVERRIDE:
        return PREPROCESS_OVERRIDE
    for layer in _walk_layers(model):
        cls = layer.__class__.__name__.lower()
        name = layer.name.lower()
        if (
            cls in ("rescaling", "normalization", "lambda")
            or "rescal" in name
            or "preprocess" in name
            or name.startswith("tf.nn.bias_add")
        ):
            return "raw"  # preprocessing is baked into the saved model
    stem = path.stem.lower()
    if "resnet" in stem or "vgg" in stem:
        return "caffe"
    return "raw"


def get_input_size(model):
    """(width, height) the model expects; falls back to 224 x 224."""
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        height, width = shape[1], shape[2]
        if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
            return (width, height)
    except Exception:
        pass
    return (224, 224)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    path = resolve_model_path()
    try:
        model = tf.keras.models.load_model(str(path), compile=False)
    except Exception:
        # Models that contain Lambda layers need safe_mode off (our own file).
        model = tf.keras.models.load_model(str(path), compile=False, safe_mode=False)
    return {
        "model": model,
        "path": path,
        "name": model_display_name(path),
        "mode": detect_preprocess_mode(model, path),
        "size": get_input_size(model),
    }


# =====================================================
# Upload decoding (all the ways a file can be bad)
# =====================================================
class InvalidUpload(Exception):
    """The file is not a usable image at all."""


def _to_rgb(img: Image.Image) -> Image.Image:
    mode = img.mode
    # 16-bit / float grayscale (common for medical exports): rescale, don't clip
    if mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        arr = np.nan_to_num(np.asarray(img).astype(np.float32))
        lo, hi = np.percentile(arr, (0.5, 99.5))
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
        mode = "L"
    # Transparent PNG/GIF/WebP: flatten onto black (MRI background), not white
    if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, rgba)
    img = img.convert("RGB")
    if max(img.size) > DISPLAY_MAX_SIDE:
        img.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE), Image.LANCZOS)
    return img


def open_image_bytes(data: bytes):
    """Decode bytes into an RGB PIL image. Returns (image, notes). Raises InvalidUpload."""
    if not data:
        raise InvalidUpload("The file is empty (0 bytes).")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise InvalidUpload(f"The file is larger than {MAX_UPLOAD_MB} MB.")

    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise InvalidUpload(
            "The file could not be read as an image. It may be corrupted, or it may "
            "not be an image at all (for example a renamed PDF or document)."
        )

    if img.format not in ALLOWED_FORMATS:
        raise InvalidUpload(f"Unsupported image format: {img.format}. Use PNG or JPG.")

    width, height = img.size
    if width * height > MAX_PIXELS:
        raise InvalidUpload(
            f"The image is too large ({width} × {height} px). Please upload a smaller version."
        )

    notes = []
    if getattr(img, "n_frames", 1) > 1:
        try:
            img.seek(0)
        except Exception:
            pass
        notes.append("This file has several frames; only the first frame is analysed.")

    try:
        img.load()
    except Exception:
        raise InvalidUpload("The image is corrupted or truncated and could not be decoded.")

    try:
        img = ImageOps.exif_transpose(img)  # respect phone-camera rotation
    except Exception:
        pass  # broken EXIF data is not a reason to reject the image

    try:
        return _to_rgb(img), notes
    except Exception:
        raise InvalidUpload("The image could not be converted to RGB.")


# =====================================================
# Brain-MRI validation (pure NumPy: no OpenCV / SciPy needed)
# =====================================================
VALIDATION_GRID = 96  # long side of the analysis grid


def _label_components(mask: np.ndarray):
    """4-connected component labelling. Returns (labels, sizes) with labels 1..n."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    values = mask.tolist()
    sizes = []
    current = 0
    for y in range(height):
        for x in range(width):
            if values[y][x] and labels[y, x] == 0:
                current += 1
                labels[y, x] = current
                stack = [(y, x)]
                count = 0
                while stack:
                    cy, cx = stack.pop()
                    count += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height and 0 <= nx < width and values[ny][nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
                sizes.append(count)
    return labels, sizes


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    labels, _ = _label_components(~mask)
    border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border_ids = [i for i in np.unique(border) if i != 0]
    outside = np.isin(labels, border_ids)
    return ~outside


def validate_mri(image: Image.Image) -> dict:
    """Heuristic check that an image looks like a single grayscale brain-MRI slice.

    Hard failures reject the image immediately. "Soft" findings only reject it when
    two or more of them occur together (a tightly cropped real MRI can trip one).
    """
    hard, soft, metrics = [], [], {}
    width, height = image.size

    if min(width, height) < MIN_SIDE:
        hard.append(f"The image is too small ({width} × {height} px) to be an MRI slice.")
    aspect = width / max(height, 1)
    metrics["aspect_ratio"] = round(aspect, 2)
    if aspect > 2.5 or aspect < 0.4:
        hard.append("The image has an extreme aspect ratio (banner, strip or long screenshot).")

    small = image.copy()
    small.thumbnail((256, 256), Image.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32)
    gray = rgb.mean(axis=-1)

    # 1. MRI slices are grayscale; photos, charts and screenshots are coloured.
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    mean_chroma = float(chroma.mean())
    coloured_fraction = float((chroma > 40).mean())
    metrics["mean_chroma"] = round(mean_chroma, 2)
    metrics["coloured_fraction"] = round(coloured_fraction, 3)
    if mean_chroma > 20 or coloured_fraction > 0.10:
        hard.append("The image is in colour. Brain MRI slices are grayscale, so this looks like a photo, graphic or screenshot.")

    # 2. Blank or nearly uniform images.
    p1, p99 = np.percentile(gray, (1, 99))
    metrics["gray_std"] = round(float(gray.std()), 2)
    if gray.std() < 10 or (p99 - p1) < 35:
        hard.append("The image is blank or almost uniform: there is no visible anatomy.")
        return _verdict(hard, soft, metrics)

    # 3. Background: MRI backgrounds are dark; documents/web pages are bright.
    gh, gw = gray.shape
    ph, pw = max(2, int(0.1 * gh)), max(2, int(0.1 * gw))
    corners = np.concatenate([
        gray[:ph, :pw].ravel(), gray[:ph, -pw:].ravel(),
        gray[-ph:, :pw].ravel(), gray[-ph:, -pw:].ravel(),
    ])
    corner_median = float(np.median(corners))
    metrics["corner_brightness"] = round(corner_median, 1)
    if corner_median > 150:
        hard.append("The background is bright. MRI scans have a dark background, so this looks like a document, drawing or photo.")
    elif corner_median > 100:
        soft.append("The background is not dark, unlike a typical MRI slice.")

    # 4. Head-shaped structure on a coarse grid.
    grid = image.convert("L")
    grid.thumbnail((VALIDATION_GRID, VALIDATION_GRID), Image.BILINEAR)
    g = np.asarray(grid, dtype=np.float32)
    threshold = max(15.0, 0.35 * float(np.percentile(g, 99)))
    mask = g > threshold
    labels, sizes = _label_components(mask)
    if not sizes:
        hard.append("No visible head or brain structure was found.")
        return _verdict(hard, soft, metrics)

    largest = labels == (int(np.argmax(sizes)) + 1)
    filled = _fill_holes(largest)
    total = filled.size
    area_ratio = float(filled.sum()) / total
    ys, xs = np.nonzero(filled)
    box_h = int(ys.max() - ys.min() + 1)
    box_w = int(xs.max() - xs.min() + 1)
    box_cover = box_h * box_w / total
    ellipse_fill = float(filled.sum()) / (np.pi / 4.0 * box_h * box_w)
    offset = float(np.hypot(xs.mean() / g.shape[1] - 0.5, ys.mean() / g.shape[0] - 0.5))
    inner_std = float(g[filled].std())
    big_objects = sum(1 for s in sizes if s >= 0.005 * total)

    metrics.update(
        area_ratio=round(area_ratio, 3),
        box_cover=round(box_cover, 3),
        ellipse_fill=round(ellipse_fill, 3),
        centre_offset=round(offset, 3),
        inner_std=round(inner_std, 1),
        separate_objects=big_objects,
    )

    if area_ratio < 0.08:
        hard.append("No dominant head-shaped region was found in the image.")
    if area_ratio > 0.97:
        soft.append("The picture has no dark background around the head, unlike a typical MRI slice.")
    if box_cover < 0.20:
        soft.append("The main object is very small compared with the whole image.")
    if ellipse_fill < 0.62 or ellipse_fill > 1.22:
        soft.append("The outline of the main object is not oval like a head or brain slice.")
    if offset > 0.25:
        soft.append("The main object is far off-centre.")
    if inner_std < 8:
        hard.append("The main object is a flat shape with no internal tissue texture (logo, shape or graphic).")
    elif inner_std < 14:
        soft.append("The main object has almost no internal tissue texture.")
    if big_objects >= 8:
        soft.append("The image contains many separate objects (typical of text, charts or screenshots).")

    return _verdict(hard, soft, metrics)


def _verdict(hard, soft, metrics) -> dict:
    valid = not hard and len(soft) < 2
    reasons = hard + (soft if not valid else [])
    return {
        "valid": valid,
        "reasons": reasons,                          # why it was rejected
        "warnings": soft if valid else [],           # passed, but worth a note
        "metrics": metrics,
    }


# =====================================================
# Preprocessing and prediction
# =====================================================
def preprocess_image(image: Image.Image, size, mode: str) -> np.ndarray:
    """RGB, model input size, float32, scaled the way the model was trained."""
    resized = image.convert("RGB").resize(size, Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if mode == "caffe":  # Keras ResNet / VGG preprocess_input
        array = array[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype=np.float32)
    elif mode == "rescale":
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def predict_tumour_probability(model, batch) -> float:
    output = np.asarray(model(batch, training=False), dtype=np.float64).reshape(-1)
    if output.size == 0 or not np.all(np.isfinite(output)):
        raise ValueError("The model returned an empty or non-finite output.")

    if output.size == 1:
        value = float(output[0])
        if value < 0.0 or value > 1.0:  # logits instead of probabilities
            value = 1.0 / (1.0 + np.exp(-value))
    elif output.size == 2:
        probs = output
        if probs.min() < 0.0 or probs.max() > 1.0 or abs(probs.sum() - 1.0) > 1e-3:
            exp = np.exp(probs - probs.max())
            probs = exp / exp.sum()
        value = float(probs[TUMOUR_INDEX])
    else:
        raise ValueError(
            f"Expected a binary output (1 sigmoid or 2 softmax units); got {output.size}."
        )
    return float(np.clip(value, 0.0, 1.0))


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
def _has_4d_output(layer) -> bool:
    if isinstance(layer, tf.keras.layers.InputLayer):
        return False
    try:
        return len(layer.output.shape) == 4
    except Exception:
        return False


def _last_4d_layer(container):
    for layer in reversed(container.layers):
        if getattr(layer, "layers", None):
            found = _last_4d_layer(layer)
            if found is not None:
                return found
        elif _has_4d_output(layer):
            return layer
    return None


def find_gradcam_target(model):
    """Last layer that still outputs a spatial feature map (batch, H, W, C).

    Works for ResNet (conv5_block3_out), EfficientNet (top_activation) and plain CNNs.
    Returns (target_layer, backbone); backbone is the nested model that owns the
    layer, or None when the layer sits directly in the outer model.
    """
    for top in reversed(model.layers):
        if getattr(top, "layers", None):
            layer = _last_4d_layer(top)
            if layer is not None:
                return layer, top
        elif _has_4d_output(top):
            return top, None
    return None, None


def build_gradcam_model(model, target_layer, backbone):
    """Model returning [target activations, final predictions].

    If the layer lives inside a nested backbone, every layer of the outer model
    is replayed in order (so rescaling, pooling, dropout and the classifier head
    are kept) and the backbone is swapped for a version that also exposes the
    target activations.
    """
    if backbone is None:
        return tf.keras.Model(
            inputs=model.inputs, outputs=[target_layer.output, model.output]
        )

    inner = tf.keras.Model(
        inputs=backbone.inputs, outputs=[target_layer.output, backbone.output]
    )

    outer_layers = [
        layer
        for layer in model.layers
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    if not any(layer is backbone for layer in outer_layers):
        raise ValueError(
            "The convolutional layer's backbone is not a direct child of the "
            "outer model; an architecture-specific Grad-CAM path is needed."
        )

    x = model.inputs[0]
    activations = None
    for layer in outer_layers:
        if layer is backbone:
            activations, x = inner(x, training=False)
        else:
            x = layer(x)

    return tf.keras.Model(inputs=model.inputs, outputs=[activations, x])


def make_gradcam_heatmap(batch, grad_model, tumour_class: bool) -> np.ndarray:
    """Normalized Grad-CAM heatmap (0-1) for the tumour or the non-tumour class."""
    image_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_activations, predictions = grad_model(image_tensor, training=False)
        if predictions.shape[-1] == 1:
            tumour_score = predictions[:, 0]
        else:
            tumour_score = predictions[:, TUMOUR_INDEX]
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


def jet_colormap(values: np.ndarray) -> np.ndarray:
    """Pure-NumPy 'jet' colormap: values in [0, 1] -> uint8 RGB.

    Replaces OpenCV so the app needs no cv2 / libGL on Streamlit Cloud.
    """
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def colorize_heatmap(heatmap: np.ndarray, size) -> np.ndarray:
    """Heatmap -> RGB colour image of the given (width, height)."""
    resized = Image.fromarray(heatmap.astype(np.float32), mode="F").resize(
        size, Image.BILINEAR
    )
    return jet_colormap(np.asarray(resized, dtype=np.float32))


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
# Analysis (stored in session_state so sliders don't reset it)
# =====================================================
def run_analysis(bundle: dict, image: Image.Image) -> dict:
    start = time.time()
    model = bundle["model"]
    batch = preprocess_image(image, bundle["size"], bundle["mode"])
    tumour_probability = predict_tumour_probability(model, batch)

    analysis = {
        "tumour_probability": tumour_probability,
        "heat_tumour": None,
        "heat_clear": None,
        "layer_name": None,
        "gradcam_error": None,
        "model_name": bundle["name"],
        "model_file": bundle["path"].name,
    }

    try:
        target_layer, backbone = find_gradcam_target(model)
        if target_layer is None:
            raise ValueError("No convolutional target layer was found.")
        grad_model = build_gradcam_model(model, target_layer, backbone)
        heat_tumour = make_gradcam_heatmap(batch, grad_model, True)
        heat_clear = make_gradcam_heatmap(batch, grad_model, False)
        if not (np.any(heat_tumour > 0) or np.any(heat_clear > 0)):
            raise ValueError(
                "The heatmap is empty (the model output is saturated for this image)."
            )
        analysis["heat_tumour"] = heat_tumour
        analysis["heat_clear"] = heat_clear
        analysis["layer_name"] = target_layer.name
    except Exception as exc:  # Grad-CAM must never break the prediction
        analysis["gradcam_error"] = str(exc)

    analysis["seconds"] = time.time() - start
    return analysis


def image_key(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.md5(rgb.tobytes())
    digest.update(str(rgb.size).encode())
    return digest.hexdigest()


def get_validation(image: Image.Image, key: str) -> dict:
    cached = st.session_state.get("validation")
    if cached and cached[0] == key:
        return cached[1]
    try:
        result = validate_mri(image)
    except Exception as exc:  # a validator bug must not take the app down
        result = {
            "valid": False,
            "reasons": [f"The image could not be checked ({exc})."],
            "warnings": [],
            "metrics": {},
        }
    st.session_state["validation"] = (key, result)
    return result


# =====================================================
# UI helpers
# =====================================================
def probability_bar(label: str, fraction: float, color: str) -> str:
    pct = fraction * 100
    return f"""
    <div class="bar-wrap">
      <div class="bar-head"><span>{label}</span><span>{pct:.2f}%</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct:.2f}%; background:{color};"></div></div>
    </div>
    """


def confidence_level(confidence: float):
    if confidence >= 0.90:
        return "High", GREEN
    if confidence >= 0.70:
        return "Moderate", AMBER
    return "Low", RED


def invalid_card(title: str, reasons) -> str:
    items = "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
    return f"""
    <div class="result-card invalid">
      <div class="result-label">Validation</div>
      <div class="result-value">❌ {html.escape(title)}</div>
      <div class="result-sub">No prediction was made.<ul>{items}</ul></div>
    </div>
    """


def load_metrics():
    for path in METRICS_CANDIDATES:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if isinstance(data, dict) and data:
                    # keep only numeric entries (bool is an int subclass, so skip it)
                    return {
                        k: float(v)
                        for k, v in data.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    } or None
            except Exception:
                return None
    return None


def format_metric(value: float) -> str:
    # accept both 0.9573 and 95.73
    return f"{value * 100:.2f}%" if value <= 1.0 else f"{value:.2f}%"


# =====================================================
# Hero + sidebar
# =====================================================
st.markdown(
    f"""
    <div class="hero">
      <h1>🧠 Brain MRI Classifier</h1>
      <p>Upload a brain MRI slice to get a Tumourous / Non-Tumourous prediction,
         class probabilities, and a Grad-CAM view of the regions that influenced the model.</p>
      <span class="pill">Deep learning</span>
      <span class="pill">{html.escape(MODEL_NAME)}</span>
      <span class="pill">Transfer learning</span>
      <span class="pill">Explainable AI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Images with a tumour probability at or above this value are labelled "
        "Tumourous. Lowering it catches more tumours (higher recall) but gives more "
        "false alarms.",
    )
    overlay_alpha = st.slider(
        "Grad-CAM overlay strength", 0.10, 0.80, 0.40, 0.05,
        help="How strongly the heatmap is blended over the MRI.",
    )
    st.divider()
    st.caption(f"Model: **{MODEL_NAME}**")
    if MODEL_PATH_PREVIEW is not None:
        st.caption(f"File: `{MODEL_PATH_PREVIEW.name}`")
    else:
        st.caption("⚠️ No model file found in `models/`")
    st.caption("Input: RGB image, resized to the model's input size")
    st.caption("Task: binary classification (Tumourous vs Non-Tumourous)")

st.warning(
    "⚠️ **Research demonstration only.** This is not a medical device and must not be "
    "used for diagnosis. MRI scans must be interpreted by a qualified radiologist.",
)

tab_analyze, tab_how, tab_about = st.tabs(["🔍 Analyze", "🧪 How it works", "ℹ️ About & limitations"])

# =====================================================
# Tab 1: Analyze
# =====================================================
with tab_analyze:
    left, right = st.columns([1, 1.15], gap="large")

    image = None
    notes = []
    upload_error = None
    validation = None
    current_key = None

    with left:
        st.subheader("1 · Choose an MRI")
        available_samples = [p for p in SAMPLE_IMAGES if p.exists()]
        options = [UPLOAD_OPTION] + [p.name for p in available_samples]
        source = st.selectbox("Image source", options)

        if source == UPLOAD_OPTION:
            uploaded = st.file_uploader(
                "Upload a brain MRI (PNG / JPG)", type=["png", "jpg", "jpeg"]
            )
            if uploaded is not None:
                try:
                    image, notes = open_image_bytes(uploaded.getvalue())
                except InvalidUpload as exc:
                    upload_error = str(exc)
                except Exception:
                    upload_error = "That file could not be read as an image."
        else:
            chosen = next((p for p in available_samples if p.name == source), None)
            if chosen is None:
                upload_error = "The selected sample image is no longer available."
            else:
                try:
                    image, notes = open_image_bytes(chosen.read_bytes())
                except Exception:
                    upload_error = "The selected sample image could not be read."

        if image is not None:
            current_key = image_key(image)
            validation = get_validation(image, current_key)
            show_image(image, caption="Selected MRI")
            for note in notes:
                st.caption(f"ℹ️ {note}")
            analyze_clicked = primary_button(
                "🔬 Analyze MRI", disabled=not validation["valid"]
            )
        else:
            analyze_clicked = False
            if upload_error:
                st.error(upload_error)
            else:
                st.info("Select a sample or upload an MRI image to begin.")

    # ---- run analysis when requested (only for validated images) ----
    model_error = None
    if image is not None and validation["valid"] and analyze_clicked:
        bundle = None
        try:
            bundle = load_model()
        except Exception as exc:
            model_error = f"Could not load the model: {exc}"
            st.session_state.pop("analysis", None)

        if bundle is not None:
            with st.spinner("Running prediction and generating Grad-CAM..."):
                try:
                    result = run_analysis(bundle, image)
                    result["key"] = current_key
                    st.session_state["analysis"] = result
                except Exception as exc:
                    model_error = f"Prediction failed: {exc}"
                    st.session_state.pop("analysis", None)

    analysis = st.session_state.get("analysis")
    has_result = (
        image is not None
        and validation is not None
        and validation["valid"]
        and analysis is not None
        and analysis.get("key") == current_key
    )

    # ---- results column ----
    with right:
        st.subheader("2 · Result")
        if upload_error:
            st.markdown(invalid_card("Invalid file", [upload_error]), unsafe_allow_html=True)
        elif validation is not None and not validation["valid"]:
            st.markdown(
                invalid_card("Invalid image: not a brain MRI", validation["reasons"]),
                unsafe_allow_html=True,
            )
            st.caption(
                "Please upload a single grayscale brain MRI slice (axial, coronal or "
                "sagittal) as PNG or JPG. Photos, screenshots, documents, charts and "
                "other body parts are rejected."
            )
        elif model_error:
            st.error(model_error)
        elif not has_result:
            st.caption("The prediction will appear here after you click **Analyze MRI**.")
            for warning in validation["warnings"] if validation else []:
                st.warning(f"Image check: {warning}")
        else:
            res = build_result(analysis["tumour_probability"], threshold)
            card_class = "tumour" if res["is_tumour"] else "clear"
            level, level_color = confidence_level(res["confidence"])

            for warning in validation["warnings"]:
                st.warning(
                    f"Image check: {warning} The result may be less reliable."
                )

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

            # Class probabilities, exactly as: Tumourous: 19.08% / Non-Tumourous: 80.92%
            st.markdown("**Class probabilities**")
            st.markdown(
                probability_bar("Tumourous", res["tumour_probability"], RED)
                + probability_bar("Non-Tumourous", res["non_tumour_probability"], GREEN),
                unsafe_allow_html=True,
            )
            st.write(f"Tumourous: {res['tumour_probability'] * 100:.2f}%")
            st.write(f"Non-Tumourous: {res['non_tumour_probability'] * 100:.2f}%")

            if abs(res["tumour_probability"] - threshold) <= 0.15:
                st.warning(
                    "The model is close to the decision boundary for this image. "
                    "Treat this prediction with extra caution."
                )

            c1, c2 = st.columns(2)
            c1.metric("Decision threshold", f"{threshold:.2f}")
            c2.metric("Inference time", f"{analysis['seconds']:.1f} s")

            st.caption(
                f"Model: {analysis['model_name']} (`{analysis['model_file']}`). "
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

            target_class = "Tumourous" if res["is_tumour"] else "Non-Tumourous"
            st.caption(
                f"Heatmap for the predicted class (**{target_class}**) · "
                f"Model: {analysis['model_name']} · Target layer: `{analysis['layer_name']}`"
            )

            g1, g2, g3 = st.columns(3)
            with g1:
                show_image(image, caption="Original MRI")
            with g2:
                show_image(colored, caption="Grad-CAM heatmap")
            with g3:
                show_image(overlay, caption="Overlay")

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
# Tab 2: How it works
# =====================================================
with tab_how:
    st.subheader("From MRI slice to prediction")
    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("1 · Validate & preprocess", "The upload is checked to be a grayscale brain-MRI-like image, then converted to RGB and resized for the model."),
        ("2 · Extract features", f"{MODEL_NAME}, pre-trained on ImageNet, extracts visual features from the MRI."),
        ("3 · Classify", "A sigmoid output gives the probability that the scan is Tumourous."),
        ("4 · Explain", "Grad-CAM highlights image regions that influenced the prediction."),
    ]
    for col, (title, text) in zip((s1, s2, s3, s4), steps):
        with col:
            st.markdown(
                f'<div class="step-card"><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown(
        """
**Input validation.** The classifier only knows two classes, so it would label *any*
picture as Tumourous or Non-Tumourous. Before predicting, the app therefore rejects files
that are unreadable, corrupted, blank, coloured, document-like, or that have no
head-shaped structure on a dark background.

**Decision threshold.** The model outputs a probability. By default, anything at or
above 0.50 is labelled *Tumourous*. In a screening-style problem, missing a tumour is
usually worse than a false alarm, so a lower threshold (the sidebar slider) trades some
precision for higher recall.

**Grad-CAM.** Gradients of the predicted class are propagated back to the last
convolutional layer. Averaging them gives importance weights for each feature map, and
the weighted sum is shown as a heatmap over the original image.
        """
    )

    metrics = load_metrics()
    if metrics:
        st.subheader("Test-set performance")
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name.replace("_", " ").title(), format_metric(value))

# =====================================================
# Tab 3: About
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
- The "is this a brain MRI?" check is a heuristic (colour, background, head-shaped
  outline). It blocks obvious non-MRI files but cannot guarantee that every accepted
  image is a real brain MRI, or that every rejected one is not.
- Grad-CAM shows model attention, not a verified lesion boundary.
- It should not be used for diagnosis, screening or any clinical decision.
        """
    )

st.markdown(
    '<div class="footer">Built with Streamlit & TensorFlow · For research and education only</div>',
    unsafe_allow_html=True,
)

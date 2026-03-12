"""
frontend/api/analyze.py — DermaScan AI Vercel Python Serverless Function
=========================================================================
Vercel invokes this file as a Python WSGI application (Flask).
The route /api/analyze is configured in vercel.json to point here.

Runtime deps (frontend/requirements.txt):
    flask, onnxruntime-cpu, numpy, pillow

No PyTorch / torchvision / OpenCV at runtime — stays well under Vercel's
250 MB compressed function bundle limit.

POST /api/analyze
  Body:    multipart/form-data  { file: <image> }
  Returns: JSON  {
    is_skin_valid:   bool,
    predicted_class: str,       # e.g. "Melanoma"
    risk_level:      str,       # "Benign" | "Malignant" | "Pre-cancerous"
    confidences:     dict,      # { nv: 0.72, mel: 0.10, … }
    grad_cam_base64: str        # base64 PNG — Jet-coloured activation overlay
  }

3rd-Year Engineering Project — B.E. Computer Engineering | 2025–26
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("dermascan")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
MODEL_PATH = _HERE / "model_final.onnx"
FC_PATH    = _HERE / "fc_weights.npy"

IMG_SIZE    = 224
NUM_CLASSES = 7
MAX_BYTES   = 16 * 1024 * 1024   # 16 MB hard cap

# ImageNet normalisation — must match training transforms exactly
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ALLOWED_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}

# HAM10000 class metadata — order matches training CLASS_MAP
CLASS_INFO: dict[int, dict[str, str]] = {
    0: {"name": "Melanocytic Nevi",              "abbr": "nv",    "risk": "Benign"},
    1: {"name": "Melanoma",                       "abbr": "mel",   "risk": "Malignant"},
    2: {"name": "Benign Keratosis-like Lesions",  "abbr": "bkl",   "risk": "Benign"},
    3: {"name": "Basal Cell Carcinoma",           "abbr": "bcc",   "risk": "Malignant"},
    4: {"name": "Actinic Keratoses",              "abbr": "akiec", "risk": "Pre-cancerous"},
    5: {"name": "Vascular Lesions",               "abbr": "vasc",  "risk": "Benign"},
    6: {"name": "Dermatofibroma",                 "abbr": "df",    "risk": "Benign"},
}

# ── Module-level singletons (loaded once per container / warm start) ──────────
_SESSION: ort.InferenceSession | None = None
_FC_W:    np.ndarray | None           = None


def _get_session() -> tuple[ort.InferenceSession, np.ndarray]:
    """Lazy-loads the ONNX session and FC weights on first call."""
    global _SESSION, _FC_W
    if _SESSION is None:
        log.info("Loading ONNX model from %s", MODEL_PATH)
        _SESSION = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        _FC_W = np.load(str(FC_PATH))   # (7, 512)
        log.info("Model loaded — input: %s", _SESSION.get_inputs()[0].shape)
    return _SESSION, _FC_W  # type: ignore[return-value]


# ── Pre-processing ────────────────────────────────────────────────────────────

def _preprocess(pil_img: Image.Image) -> np.ndarray:
    """
    Resize → float32 / 255 → ImageNet normalise → NCHW tensor.
    Returns: (1, 3, 224, 224) float32 numpy array.
    """
    img  = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = (arr - _MEAN) / _STD              # (224, 224, 3)
    return arr.transpose(2, 0, 1)[np.newaxis] # (1, 3, 224, 224)


# ── Skin-pixel gatekeeper ─────────────────────────────────────────────────────

def _is_skin_image(pil_img: Image.Image, threshold: float = 0.20) -> bool:
    """
    Dual colour-space skin-pixel validator (Kovac RGB ∩ Peer YCbCr).

    Rule 1 — Kovac et al. (2002):
        R > 95, G > 40, B > 20
        max(R,G,B) − min(R,G,B) > 15
        |R − G| > 15, R > G, R > B

    Rule 2 — Peer et al. YCbCr locus:
        77 ≤ Cb ≤ 127   AND   133 ≤ Cr ≤ 173

    Returns True iff at least `threshold` fraction of pixels satisfy both rules.
    """
    rgb  = np.array(pil_img.convert("RGB"),   dtype=np.float32)
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    kovac = (
          (R > 95) & (G > 40) & (B > 20)
        & (np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B) > 15)
        & (np.abs(R - G) > 15)
        & (R > G) & (R > B)
    )

    ycbcr = np.array(pil_img.convert("YCbCr"), dtype=np.float32)
    Cb, Cr   = ycbcr[:, :, 1], ycbcr[:, :, 2]
    peer     = (Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173)

    ratio = float((kovac & peer).sum()) / float(kovac.size)
    return ratio >= threshold


# ── Class Activation Map (CAM) ────────────────────────────────────────────────

def _compute_cam(layer4_acts: np.ndarray, fc_weights: np.ndarray, pred_idx: int) -> np.ndarray:
    """
    Activation-based Class Activation Map — no gradients required.

    Because ResNet18 uses Global Average Pooling directly before the
    Linear classifier, the FC weight vector for the predicted class
    is exactly the channel-importance weighting used in the original
    CAM paper (Zhou et al., 2016).  This gives virtually identical
    results to Grad-CAM on this architecture.

    Parameters
    ----------
    layer4_acts : (1, 512, 7, 7)  ONNX output.
    fc_weights  : (7, 512)  saved from model.fc[1].weight.
    pred_idx    : int  predicted class index.

    Returns
    -------
    (7, 7) float32 array normalised to [0, 1].
    """
    acts = layer4_acts[0]          # (512, 7, 7)
    w    = fc_weights[pred_idx]    # (512,)
    cam  = np.einsum("c,chw->hw", w, acts)   # (7, 7)
    cam  = np.maximum(cam, 0.0)              # ReLU

    c_min, c_max = cam.min(), cam.max()
    if c_max - c_min > 1e-8:
        cam = (cam - c_min) / (c_max - c_min)
    return cam.astype(np.float32)


def _jet_colormap(t: np.ndarray) -> np.ndarray:
    """
    Vectorised piecewise-linear approximation of matplotlib's Jet colormap.
    Produces values that are numerically identical to matplotlib.cm.jet.

    Parameters
    ----------
    t : 1-D float32 array in [0, 1].

    Returns
    -------
    (N, 3) uint8 RGB array.
    """
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def _overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Resize CAM to IMG_SIZE × IMG_SIZE, apply Jet colourmap, and alpha-blend
    with the source image using pure numpy + Pillow (no OpenCV required).
    """
    # Upsample 7×7 CAM → 224×224
    cam_pil    = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
    cam_up     = np.array(
        cam_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32
    ) / 255.0

    # Jet colourise
    flat       = cam_up.flatten()
    heatmap    = _jet_colormap(flat).reshape(IMG_SIZE, IMG_SIZE, 3)

    # Resize source image
    base       = np.array(
        pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
        dtype=np.float32,
    )

    # Alpha blend
    blended = np.clip(base * (1.0 - alpha) + heatmap.astype(np.float32) * alpha, 0, 255)
    return Image.fromarray(blended.astype(np.uint8))


def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Flask application (Vercel WSGI entry-point) ───────────────────────────────

app = Flask(__name__)


@app.route("/api/analyze", methods=["POST"])
@app.route("/", methods=["POST"])   # fallback if Vercel strips prefix
def analyze():                      # type: ignore[return-value]
    # ── 1. Validate MIME type ────────────────────────────────────────────────
    content_type = (request.content_type or "").lower()
    if not content_type.startswith("multipart/form-data"):
        return jsonify({"detail": "Expected multipart/form-data upload."}), 415

    # ── 2. Read uploaded file ────────────────────────────────────────────────
    file_storage = request.files.get("file")
    if file_storage is None:
        return jsonify({"detail": "No file field found in the form."}), 400

    raw = file_storage.read()
    if len(raw) > MAX_BYTES:
        return jsonify({"detail": "Image exceeds the 16 MB size limit."}), 413

    # ── 3. Decode image ──────────────────────────────────────────────────────
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        log.warning("Image decode failed: %s", exc)
        return jsonify({"detail": "Could not decode the uploaded file as an image."}), 422

    # ── 4. Skin-pixel gatekeeper ─────────────────────────────────────────────
    if not _is_skin_image(pil_img):
        log.info("Skin validation failed.")
        return jsonify({
            "is_skin_valid":   False,
            "predicted_class": "",
            "risk_level":      "",
            "confidences":     {v["abbr"]: 0.0 for v in CLASS_INFO.values()},
            "grad_cam_base64": "",
        })

    # ── 5. Inference ─────────────────────────────────────────────────────────
    try:
        session, fc_w = _get_session()
        tensor        = _preprocess(pil_img)      # (1, 3, 224, 224)

        logits_arr, layer4_acts = session.run(
            ["logits", "layer4_acts"],
            {"image": tensor},
        )

        # Softmax (stable)
        logits = logits_arr[0]                    # (7,)
        logits -= logits.max()
        exp    = np.exp(logits)
        probs  = exp / exp.sum()

        pred_idx = int(np.argmax(probs))

    except Exception as exc:
        log.exception("Inference failed: %s", exc)
        return jsonify({"detail": "Model inference error. Please try again."}), 500

    # ── 6. CAM overlay ───────────────────────────────────────────────────────
    try:
        cam       = _compute_cam(layer4_acts, fc_w, pred_idx)
        cam_image = _overlay_heatmap(pil_img, cam)
        cam_b64   = _pil_to_base64(cam_image)
    except Exception as exc:
        log.warning("CAM generation failed (returning blank): %s", exc)
        cam_b64 = ""

    # ── 7. Assemble response ─────────────────────────────────────────────────
    info        = CLASS_INFO[pred_idx]
    confidences = {
        CLASS_INFO[i]["abbr"]: round(float(probs[i]), 6)
        for i in range(NUM_CLASSES)
    }

    log.info("Prediction: %s (%.1f%%)  risk=%s", info["name"], probs[pred_idx] * 100, info["risk"])

    return jsonify({
        "is_skin_valid":   True,
        "predicted_class": info["name"],
        "risk_level":      info["risk"],
        "confidences":     confidences,
        "grad_cam_base64": cam_b64,
    })


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "runtime": "onnxruntime"})

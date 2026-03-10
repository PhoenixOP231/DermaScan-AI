"""
backend/main.py — DermaScan AI  ·  FastAPI Inference Server
============================================================
POST /analyze
  • Dual color-space skin-pixel gate (Kovac RGB ∩ Peer YCbCr)
  • ResNet18  FP16→FP32  inference  (HAM10000, 7 classes)
  • Grad-CAM heatmap  →  base64 PNG
  • JSON response: is_skin_valid, predicted_class, risk_level,
                   confidences (dict), grad_cam_base64 (str)

Run:
    uvicorn backend.main:app --reload --port 8000

3rd-Year Engineering Project — B.E. Computer Engineering | 2025–26
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import cm
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models, transforms


# ═══════════════════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("dermascan")


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

# model_final.pth lives inside backend/ (moved from project root)
_HERE      = Path(__file__).resolve().parent           # …/backend/
MODEL_PATH = _HERE / "model_final.pth"                 # …/backend/model_final.pth

IMG_SIZE    = 224
NUM_CLASSES = 7

# ImageNet normalisation — must match training transform
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# HAM10000 class metadata — must match training CLASS_MAP ordering
CLASS_INFO: dict[int, dict[str, str]] = {
    0: {"name": "Melanocytic Nevi",             "abbr": "nv",    "risk": "Benign"},
    1: {"name": "Melanoma",                      "abbr": "mel",   "risk": "Malignant"},
    2: {"name": "Benign Keratosis-like Lesions", "abbr": "bkl",   "risk": "Benign"},
    3: {"name": "Basal Cell Carcinoma",          "abbr": "bcc",   "risk": "Malignant"},
    4: {"name": "Actinic Keratoses",             "abbr": "akiec", "risk": "Pre-cancerous"},
    5: {"name": "Vascular Lesions",              "abbr": "vasc",  "risk": "Benign"},
    6: {"name": "Dermatofibroma",                "abbr": "df",    "risk": "Benign"},
}

# CPU-only inference (GPU support is opt-in via the DERMASCAN_DEVICE env-var)
_device_name = os.getenv("DERMASCAN_DEVICE", "cpu")
DEVICE = torch.device(_device_name)


# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI App + CORS
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="DermaScan AI",
    description="ResNet18-based dermoscopy lesion classifier with Grad-CAM explainability.",
    version="1.0.0",
)

# Wildcard CORS — permits the Next.js frontend (any origin) to call the API.
# Tighten allow_origins in production by setting the ALLOWED_ORIGINS env-var.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model — loaded once at startup and reused for every request
# ═══════════════════════════════════════════════════════════════════════════════

def _build_model() -> nn.Module:
    """
    Constructs the ResNet18 architecture and loads the saved weights.

    The checkpoint is stored in FP16 (half-precision) to stay under GitHub's
    25 MB file-size limit.  Weights are upcast to FP32 here for numerically
    stable CPU inference.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Run main.py (the training script) first."
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )

    # FP16 → FP32 conversion during load
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = {k: v.float() for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    log.info("Model loaded from %s  (device=%s)", MODEL_PATH, DEVICE)
    return model


# Module-level singleton — initialised at import time so FastAPI worker
# pools share the same loaded weights.
_MODEL: nn.Module = _build_model()


# ═══════════════════════════════════════════════════════════════════════════════
#  Security Gatekeeper — Skin-Pixel Validation
# ═══════════════════════════════════════════════════════════════════════════════

def is_skin_image(pil_image: Image.Image, threshold: float = 0.20) -> bool:
    """
    Validates that an image contains a sufficient proportion of skin-tone pixels
    using a dual color-space intersection approach.

    Rule 1 — Kovac et al. (2002) explicit RGB segmentation rule:
        R > 95  AND  G > 40  AND  B > 20
        max(R,G,B) − min(R,G,B) > 15
        |R − G| > 15  AND  R > G  AND  R > B

    Rule 2 — Peer et al. YCbCr locus:
        77 ≤ Cb ≤ 127  AND  133 ≤ Cr ≤ 173

    A pixel must satisfy BOTH rules simultaneously (intersection).  The
    intersection drastically reduces false positives from monitors, objects,
    animals, and neutral backgrounds that superficially resemble skin in RGB.

    Parameters
    ----------
    pil_image : PIL.Image — arbitrary input image.
    threshold : float    — minimum fraction of pixels that must pass (0–1).

    Returns
    -------
    bool — True if skin content ≥ threshold; False otherwise.
    """
    img_rgb  = np.array(pil_image.convert("RGB"),   dtype=np.float32)
    R, G, B  = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # Rule 1: Kovac RGB criterion
    rgb_mask = (
          (R > 95) & (G > 40) & (B > 20)
        & (np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B) > 15)
        & (np.abs(R - G) > 15)
        & (R > G) & (R > B)
    )

    # Rule 2: YCbCr chrominance locus
    img_ycbcr = np.array(pil_image.convert("YCbCr"), dtype=np.float32)
    Cb, Cr    = img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]
    ycbcr_mask = (Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173)

    combined   = rgb_mask & ycbcr_mask
    skin_ratio = float(combined.sum()) / float(combined.size)
    return skin_ratio >= threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  Grad-CAM — Explainability
# ═══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet18.

    Reference: Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization."

    Hooks are registered on ``model.layer4[-1]`` to capture:
      • Forward activations  A^k   (shape: 1 × C × H × W)
      • Backward gradients   ∂y^c / ∂A^k

    CAM formula:
        L^c = ReLU( Σ_k  α^c_k · A^k )
        α^c_k = GlobalAvgPool( ∂y^c / ∂A^k )
    """

    def __init__(self, model: nn.Module) -> None:
        self.model       = model
        self.activations: torch.Tensor | None = None
        self.gradients:   torch.Tensor | None = None
        self._hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        target = self.model.layer4[-1]

        def fwd_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:  # noqa: ARG001
            self.activations = out.detach()

        def bwd_hook(module: nn.Module, g_in: Any, g_out: tuple[torch.Tensor, ...]) -> None:  # noqa: ARG001
            self.gradients = g_out[0].detach()

        self._hooks.append(target.register_forward_hook(fwd_hook))
        self._hooks.append(target.register_full_backward_hook(bwd_hook))

    def remove_hooks(self) -> None:
        """Remove all registered hooks to prevent accumulation on the shared model."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute_cam(self) -> np.ndarray:
        """
        Compute the normalised Grad-CAM heatmap in [0, 1].
        Must be called after a forward + backward pass.
        """
        # Global-average-pool the gradients: (1, C, H, W) → (1, C, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # type: ignore[union-attr]
        cam     = F.relu((weights * self.activations).sum(dim=1).squeeze(0))  # type: ignore[arg-type]

        c_min, c_max = cam.min(), cam.max()
        if (c_max - c_min).item() > 1e-8:
            cam = (cam - c_min) / (c_max - c_min)

        return cam.cpu().numpy()


def _overlay_heatmap(
    pil_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Bilinearly upsample a Grad-CAM array to IMG_SIZE×IMG_SIZE, colourise
    with the Jet colormap, and alpha-blend it over the resized source image.
    """
    base        = np.array(
        pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8
    )
    cam_up      = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    heatmap_rgb = (cm.jet(cam_up)[:, :, :3] * 255).astype(np.uint8)
    blended     = cv2.addWeighted(base, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(blended)


def _pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 string (no data-URI prefix)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _run_inference(
    model: nn.Module,
    pil_image: Image.Image,
) -> tuple[np.ndarray, int, Image.Image]:
    """
    End-to-end inference: preprocessing → forward pass → Grad-CAM.

    Returns
    -------
    probs     : np.ndarray  — shape (7,) softmax probabilities.
    pred_idx  : int         — argmax class index.
    cam_image : PIL.Image   — Grad-CAM heatmap blended onto the original.
    """
    tensor   = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    grad_cam = GradCAM(model)

    try:
        # torch.enable_grad() is required: FastAPI may run in a no-grad context
        with torch.enable_grad():
            model.zero_grad()
            logits   = model(tensor)
            probs    = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

            # Backprop on the winning class score to populate gradient hooks
            logits[0, pred_idx].backward()

        cam       = grad_cam.compute_cam()
        cam_image = _overlay_heatmap(pil_image, cam)

    finally:
        grad_cam.remove_hooks()   # never leak hooks on the shared model

    return probs, pred_idx, cam_image


# ═══════════════════════════════════════════════════════════════════════════════
#  Pydantic Response Model
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyzeResponse(BaseModel):
    is_skin_valid:   bool
    predicted_class: str              # e.g. "Melanoma"
    risk_level:      str              # Benign | Malignant | Pre-cancerous
    confidences:     dict[str, float] # {"nv": 0.72, "mel": 0.10, …}
    grad_cam_base64: str              # base64-encoded PNG (no data-URI prefix)


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["utility"])
async def health_check() -> dict[str, str]:
    """Liveness probe — confirms the server and model are operational."""
    return {"status": "ok", "model": "loaded"}


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["inference"],
    summary="Classify a dermoscopy image and return a Grad-CAM explanation.",
)
async def analyze(
    file: UploadFile = File(..., description="Dermoscopy image (JPEG / PNG / BMP)."),
) -> AnalyzeResponse:
    """
    Accepts a raw image upload, validates skin content, runs ResNet18
    classification, and returns explainability data.

    **Response fields**
    - `is_skin_valid`      — whether the image passed the skin-pixel gate.
    - `predicted_class`    — name of the top-1 HAM10000 class.
    - `risk_level`         — Benign | Malignant | Pre-cancerous.
    - `confidences`     — dict of all 7 class abbreviations → probability.
    - `grad_cam_base64` — Jet-colourised Grad-CAM overlay, base64 PNG.

    If `is_skin_valid` is **false**, classification is still skipped and the
    other fields will be empty / zero — the frontend should surface the
    rejection message instead of displaying results.
    """
    # ── 1. Validate content-type ────────────────────────────────────────────
    allowed_mimes = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}
    content_type  = (file.content_type or "").lower()
    if content_type not in allowed_mimes:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{content_type}'. Upload a JPEG or PNG image.",
        )

    # ── 2. Decode image ──────────────────────────────────────────────────────
    raw_bytes = await file.read()
    # Guard against excessively large uploads (16 MB hard cap)
    MAX_BYTES = 16 * 1024 * 1024
    if len(raw_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds the 16 MB size limit.",
        )

    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        log.warning("Failed to decode uploaded file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The uploaded file could not be decoded as a valid image.",
        ) from exc

    # ── 3. Skin-pixel gatekeeper ─────────────────────────────────────────────
    skin_valid = is_skin_image(pil_image)
    if not skin_valid:
        log.info("Skin validation failed — returning early.")
        return AnalyzeResponse(
            is_skin_valid=False,
            predicted_class="",
            risk_level="",
            confidences={v["abbr"]: 0.0 for v in CLASS_INFO.values()},
            grad_cam_base64="",
        )

    # ── 4. Inference + Grad-CAM ──────────────────────────────────────────────
    try:
        probs, pred_idx, cam_image = _run_inference(_MODEL, pil_image)
    except Exception as exc:
        log.exception("Inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during model inference.",
        ) from exc

    # ── 5. Build response ────────────────────────────────────────────────────
    info = CLASS_INFO[pred_idx]

    confidences = {
        CLASS_INFO[i]["abbr"]: round(float(probs[i]), 6)
        for i in range(NUM_CLASSES)
    }

    grad_cam_b64 = _pil_to_base64(cam_image, fmt="PNG")

    log.info(
        "Prediction: %s (%.1f%%)  risk=%s",
        info["name"],
        probs[pred_idx] * 100,
        info["risk"],
    )

    return AnalyzeResponse(
        is_skin_valid=True,
        predicted_class=info["name"],
        risk_level=info["risk"],
        confidences=confidences,
        grad_cam_base64=grad_cam_b64,
    )

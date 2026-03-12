"""
app.py — DermaScan AI Streamlit Web Application
================================================
AI-powered skin cancer detection with:
  • Skin-pixel gatekeeper (Kovac et al. RGB rules)
  • Grad-CAM explainability heatmap overlay
  • Auto dark / light mode via CSS custom properties
  • Fully mobile-responsive (Android · iOS · tablet · desktop)
  • Premium UI: Inter + JetBrains Mono fonts, SVG logo, glassmorphism cards

3rd-Year Engineering Project — B.E. Computer Engineering | 2025–26
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.cm as cm
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
#  Page Config  (must be the very first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title            = "DermaScan AI",
    page_icon             = "🔬",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Custom CSS — Dark-mode Premium UI
# ═══════════════════════════════════════════════════════════════════════════════

_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Design tokens ──────────────────────────────────────────────────────── */
:root {
    --bg-app      : #f0f4f8;
    --bg-card     : #ffffff;
    --border      : #e2e8f0;
    --border-hi   : #93c5fd;
    --text-primary: #0f172a;
    --text-second : #334155;
    --text-muted  : #64748b;
    --text-faint  : #94a3b8;
    --accent      : #2563eb;
    --accent-dark : #1d4ed8;
    --accent-light: #eff6ff;
    --success     : #16a34a;
    --success-bg  : #f0fdf4;
    --warning     : #92400e;
    --warning-bg  : #fffbeb;
    --warning-bdr : #fde68a;
    --danger      : #dc2626;
    --danger-bg   : #fef2f2;
    --shadow-sm   : 0 1px 3px rgba(15,23,42,0.07), 0 1px 2px rgba(15,23,42,0.04);
    --shadow-md   : 0 4px 14px rgba(15,23,42,0.08);
    --r-sm        : 8px;
    --r-md        : 12px;
    --r-lg        : 16px;
}

/* ─── Base ───────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background : var(--bg-app) !important;
    color      : var(--text-second) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size  : 14.5px;
    line-height: 1.6;
}
.block-container {
    padding-top   : 4rem    !important;
    padding-left  : 2rem    !important;
    padding-right : 2rem    !important;
    max-width     : 1200px  !important;
}

/* ─── Hide Streamlit top toolbar decoration ──────────────────────────────── */
[data-testid="stHeader"] {
    background: var(--bg-app) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ─── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background  : var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-second) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

/* ─── Page header ────────────────────────────────────────────────────────── */
.page-header {
    display       : flex;
    align-items   : center;
    gap           : 0.85rem;
    padding-bottom: 1.4rem;
    margin-bottom : 1.5rem;
    border-bottom : 2px solid var(--border);
}
.page-header-text { display: flex; flex-direction: column; gap: 0.1rem; }
.page-title {
    font-size     : 1.9rem;
    font-weight   : 800;
    color         : var(--text-primary) !important;
    letter-spacing: -0.03em;
    margin        : 0;
    line-height   : 1.2;
}
.page-meta {
    font-size : 0.87rem;
    color     : var(--text-muted) !important;
    font-weight: 400;
    margin    : 0;
}

/* ─── Badges / chips ──────────────────────────────────────────────────────── */
.chip {
    display      : inline-flex;
    align-items  : center;
    padding      : 0.22rem 0.65rem;
    border-radius: 99px;
    font-size    : 0.7rem;
    font-weight  : 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.chip-blue  { background: var(--accent-light); color: var(--accent);  border: 1px solid #bfdbfe; }
.chip-green { background: var(--success-bg);   color: var(--success); border: 1px solid #bbf7d0; }
.chip-amber { background: var(--warning-bg);   color: #b45309;        border: 1px solid var(--warning-bdr); }
.chip-red   { background: var(--danger-bg);    color: var(--danger);  border: 1px solid #fecaca; }

/* ─── Section label ──────────────────────────────────────────────────────── */
.hc-label {
    font-size     : 0.68rem;
    font-weight   : 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color         : var(--text-muted) !important;
    margin-bottom : 0.8rem;
}

/* ─── Cards ──────────────────────────────────────────────────────────────── */
.hc-card {
    background   : var(--bg-card);
    border       : 1px solid var(--border);
    border-radius: var(--r-lg);
    padding      : 1.5rem;
    box-shadow   : var(--shadow-sm);
    margin-bottom: 1rem;
}

/* ─── Metric tiles ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background   : var(--bg-card) !important;
    border       : 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    padding      : 1.1rem 1.25rem !important;
    box-shadow   : var(--shadow-sm) !important;
}
[data-testid="stMetricLabel"] * {
    font-size     : 0.68rem !important;
    font-weight   : 600 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color         : var(--text-muted) !important;
    font-family   : 'Inter', sans-serif !important;
}
[data-testid="stMetricValue"] * {
    font-size  : 1.3rem !important;
    font-weight: 700 !important;
    color      : var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ─── Progress bars ──────────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background   : var(--accent) !important;
    border-radius: 99px !important;
}
.stProgress > div > div > div {
    background   : #e2e8f0 !important;
    border-radius: 99px !important;
}

/* ─── Confidence rows ────────────────────────────────────────────────────── */
.conf-row {
    display        : flex;
    align-items    : center;
    justify-content: space-between;
    padding        : 0.45rem 0;
    border-bottom  : 1px solid var(--border);
    font-size      : 0.84rem;
}
.conf-row:last-child { border-bottom: none; }
.conf-label { font-weight: 500; color: var(--text-second); }
.conf-pct   { font-weight: 700; color: var(--accent); min-width: 44px; text-align: right; }

/* ─── File uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] section {
    border       : 2px dashed var(--border-hi) !important;
    border-radius: var(--r-md) !important;
    background   : var(--accent-light) !important;
    min-height   : 90px;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important;
    background  : #dbeafe !important;
}

/* ─── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    background   : var(--accent) !important;
    color        : #ffffff !important;
    border       : none !important;
    border-radius: var(--r-sm) !important;
    padding      : 0.6rem 1.5rem !important;
    font-family  : 'Inter', sans-serif !important;
    font-weight  : 600 !important;
    font-size    : 0.88rem !important;
    transition   : background 0.15s, box-shadow 0.15s !important;
    box-shadow   : 0 1px 4px rgba(37,99,235,0.25) !important;
    width        : 100% !important;
}
.stButton > button:hover {
    background: var(--accent-dark) !important;
    box-shadow: 0 3px 10px rgba(37,99,235,0.35) !important;
}

/* ─── Alerts ─────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: var(--r-md) !important; }

/* ─── Images ─────────────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: var(--r-md);
    border       : 1px solid var(--border);
    max-width    : 100%;
}

/* ─── Medical disclaimer bar ─────────────────────────────────────────────── */
.disclaimer {
    background   : var(--warning-bg);
    border       : 1px solid var(--warning-bdr);
    border-left  : 4px solid #f59e0b;
    border-radius: var(--r-md);
    padding      : 0.85rem 1.15rem;
    font-size    : 0.84rem;
    color        : var(--warning) !important;
    margin-bottom: 1.5rem;
    line-height  : 1.65;
}

/* ─── Result card ────────────────────────────────────────────────────────── */
.result-card {
    background   : var(--bg-card);
    border       : 1px solid var(--border);
    border-radius: var(--r-lg);
    padding      : 1.5rem 1.75rem;
    box-shadow   : var(--shadow-sm);
    margin-top   : 1rem;
}
.result-title { font-size: 1.1rem; font-weight: 700; margin: 0; color: var(--text-primary) !important; }
.result-body  { font-size: 0.85rem; color: var(--text-muted) !important; line-height: 1.75; margin: 0.6rem 0 0 0; }
.result-warn  { font-size: 0.82rem; color: #b45309 !important; font-weight: 500; margin-top: 0.65rem; }

/* ─── Footer ─────────────────────────────────────────────────────────────── */
.hc-footer {
    margin-top   : 2.5rem;
    padding      : 1.1rem 1rem;
    text-align   : center;
    border-top   : 1px solid var(--border);
    background   : var(--bg-card);
    border-radius: var(--r-md);
    color        : var(--text-faint) !important;
    font-size    : 0.78rem;
    line-height  : 1.9;
}

/* ─── Sidebar brand ──────────────────────────────────────────────────────── */
.sb-brand-row  { display: flex; align-items: center; gap: 0.7rem; padding: 0.35rem 0 0.1rem; }
.sb-brand-name { font-size: 1.05rem; font-weight: 800; color: var(--text-primary) !important; letter-spacing: -0.02em; }
.sb-tagline    { font-size: 0.72rem; color: var(--text-muted) !important; margin-top: 0.05rem; }
.sb-section    { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
                 color: var(--text-muted) !important; margin: 0.9rem 0 0.45rem; }
.sb-row {
    display        : flex;
    justify-content: space-between;
    align-items    : center;
    padding        : 0.38rem 0;
    border-bottom  : 1px solid var(--border);
    font-size      : 0.82rem;
}
.sb-row:last-child { border-bottom: none; }
.sb-key { color: var(--text-muted) !important; }
.sb-val { font-weight: 600; color: var(--text-primary) !important; }
.sb-step-row {
    display    : flex;
    align-items: flex-start;
    gap        : 0.55rem;
    padding    : 0.32rem 0;
    font-size  : 0.82rem;
    color      : var(--text-second);
}
.sb-step-num {
    min-width      : 20px;
    height         : 20px;
    background     : var(--accent);
    color          : white !important;
    border-radius  : 50%;
    display        : flex;
    align-items    : center;
    justify-content: center;
    font-size      : 0.62rem;
    font-weight    : 700;
    flex-shrink    : 0;
    margin-top     : 0.12rem;
}
.sb-class-row {
    display      : flex;
    align-items  : center;
    gap          : 0.45rem;
    padding      : 0.35rem 0;
    font-size    : 0.82rem;
    border-bottom: 1px solid var(--border);
}
.sb-class-row:last-child { border-bottom: none; }
.sb-abbr { font-weight: 700; min-width: 42px; color: var(--accent); }
.sb-benign    { color: var(--success); font-weight: 600; font-size: 0.75rem; }
.sb-pre       { color: #b45309;        font-weight: 600; font-size: 0.75rem; }
.sb-malignant { color: var(--danger);  font-weight: 600; font-size: 0.75rem; }

/* ─── Divider ────────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ─── Caption ────────────────────────────────────────────────────────────── */
.stCaption, small { font-family: 'Inter', sans-serif !important; color: var(--text-muted) !important; font-size: 0.78rem !important; }

/* ─── Typography ─────────────────────────────────────────────────────────── */
h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; color: var(--text-primary) !important; letter-spacing: -0.02em; font-weight: 700; }

/* ─── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-app); }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }

/* ─── Mobile ─────────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
    .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    .page-title { font-size: 1.45rem !important; }
}
@media (max-width: 480px) {
    .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    .page-title { font-size: 1.2rem !important; }
    .hc-card, .result-card { padding: 1rem; }
}
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH  = "backend/model_final.pth"
IMG_SIZE    = 224
NUM_CLASSES = 7

CLASS_INFO = {
    0: {"name": "Melanocytic Nevi",             "abbr": "NV",    "risk": "Benign",        "icon": "🟢"},
    1: {"name": "Melanoma",                      "abbr": "MEL",   "risk": "Malignant",     "icon": "🔴"},
    2: {"name": "Benign Keratosis-like Lesions", "abbr": "BKL",   "risk": "Benign",        "icon": "🟢"},
    3: {"name": "Basal Cell Carcinoma",          "abbr": "BCC",   "risk": "Malignant",     "icon": "🔴"},
    4: {"name": "Actinic Keratoses",             "abbr": "AKIEC", "risk": "Pre-cancerous", "icon": "🟡"},
    5: {"name": "Vascular Lesions",              "abbr": "VASC",  "risk": "Benign",        "icon": "🟢"},
    6: {"name": "Dermatofibroma",                "abbr": "DF",    "risk": "Benign",        "icon": "🟢"},
}

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# Streamlit Cloud inference runs on CPU only
device = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Loader
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⚙️  Loading DermaScan AI model…")
def load_model() -> nn.Module:
    """
    Loads the trained ResNet18 model from disk.

    The saved weights are stored in FP16 (half precision) to remain under
    GitHub's 25 MB per-file limit.  They are converted back to FP32 here
    for numerically stable CPU inference.

    Returns
    -------
    nn.Module
        ResNet18 in evaluation mode, ready for inference.

    Raises
    ------
    FileNotFoundError
        If model_final.pth is absent (training has not been run yet).
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    # Load FP16 state dict, convert weights to FP32 for stable CPU maths
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    state_dict = {k: v.float() for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Security Gatekeeper — Skin-Pixel Validation
# ═══════════════════════════════════════════════════════════════════════════════

def is_skin_image(pil_image: Image.Image, threshold: float = 0.20) -> bool:
    """
    Verifies that the uploaded image contains a sufficient proportion of
    skin-tone pixels before allowing model inference.

    Uses a dual color-space approach — both conditions must be satisfied
    simultaneously (intersection), which dramatically reduces false positives
    from monitors, objects, animals, and plain backgrounds.

    Rule 1 — Kovac et al. (2002) explicit RGB skin segmentation:
        R > 95  AND  G > 40  AND  B > 20
        max(R,G,B) − min(R,G,B) > 15
        |R − G| > 15  AND  R > G  AND  R > B

    Rule 2 — Peer et al. YCbCr skin range (Cb in [77,127], Cr in [133,173]):
        A pixel's chrominance channels must fall within the skin locus
        in the YCbCr space.  Screens, paper, and most non-skin objects
        emit/reflect blue-shifted or neutral light that falls outside
        this locus and therefore fail this rule even when their RGB
        values superficially resemble skin tones.

    Parameters
    ----------
    pil_image : PIL.Image
        The user-uploaded image (any size, any mode).
    threshold : float
        Minimum fraction of pixels (0–1) that must satisfy BOTH rules.
        Default raised to 0.20 (20%) for stricter real-world rejection.

    Returns
    -------
    bool
        True  — image passes the dual skin-content test.
        False — rejection; caller should display an error to the user.
    """
    img_rgb = np.array(pil_image.convert("RGB"), dtype=np.float32)
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # ── Rule 1: Kovac RGB ────────────────────────────────────────────────────
    rgb_mask = (
          (R > 95) & (G > 40) & (B > 20)
        & (  np.maximum(np.maximum(R, G), B)
           - np.minimum(np.minimum(R, G), B) > 15)
        & (np.abs(R - G) > 15)
        & (R > G) & (R > B)
    )

    # ── Rule 2: YCbCr chrominance locus ─────────────────────────────────────
    img_ycbcr = np.array(pil_image.convert("YCbCr"), dtype=np.float32)
    Cb, Cr    = img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]
    ycbcr_mask = (Cb >= 77) & (Cb <= 127) & (Cr >= 133) & (Cr <= 173)

    # ── Intersection: pixel must pass BOTH rules ─────────────────────────────
    combined_mask = rgb_mask & ycbcr_mask
    skin_ratio    = float(combined_mask.sum()) / float(combined_mask.size)
    return skin_ratio >= threshold


# ═══════════════════════════════════════════════════════════════════════════════
#  Grad-CAM — Explainable AI
# ═══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet18.

    Reference: Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization."

    Hooks are registered on ``model.layer4[-1]`` (the final residual block)
    to capture:
      • Forward activations  A^k  (feature maps,  shape: 1 × C × H × W)
      • Backward gradients   ∂y^c / ∂A^k

    The CAM is computed as:
        L^c = ReLU( Σ_k  α^c_k ⊙ A^k )
    where
        α^c_k = GlobalAvgPool( ∂y^c / ∂A^k )

    Hooks are stored and can be removed after use via ``remove_hooks()``
    to prevent accumulation on the cached model instance.
    """

    def __init__(self, model: nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.layer4[-1]

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, g_in, g_out):
            self.gradients = g_out[0].detach()

        self._hooks.append(target.register_forward_hook(fwd_hook))
        self._hooks.append(target.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        """
        Removes all registered forward and backward hooks.

        Must be called after each prediction to prevent hook accumulation
        on the shared, cached model object.
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute_cam(self) -> np.ndarray:
        """
        Computes the Grad-CAM heatmap from stored activations and gradients.

        Called AFTER a forward + backward pass has been executed.

        Returns
        -------
        np.ndarray
            2-D normalised heatmap in [0, 1], shape (H_feat, W_feat).
            Typical size for ResNet18 with 224×224 input: 7 × 7.
        """
        # α^c_k : global-average-pool gradients across spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted-sum of activation maps + ReLU (keep only positive influence)
        cam = F.relu((weights * self.activations).sum(dim=1).squeeze(0))

        # Normalise to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if (c_max - c_min).item() > 1e-8:
            cam = (cam - c_min) / (c_max - c_min)

        return cam.cpu().numpy()


def overlay_heatmap(
    pil_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Alpha-blends the Grad-CAM heatmap onto the original lesion image.

    The raw CAM (typically 7×7) is bilinearly upsampled to IMG_SIZE×IMG_SIZE,
    colourised with the Jet colormap (blue → green → yellow → red), and
    additively blended over the resized original image.

    Warm colours (red/yellow) indicate image regions that strongly increased
    the predicted class score; cool colours (blue) indicate negligible regions.

    Parameters
    ----------
    pil_image : PIL.Image
        Original uploaded image.
    cam : np.ndarray
        Grad-CAM array in [0, 1].
    alpha : float
        Heatmap opacity  (0 = invisible, 1 = fully opaque heatmap).

    Returns
    -------
    PIL.Image
        Blended RGB image at IMG_SIZE × IMG_SIZE pixels.
    """
    base = np.array(
        pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
        dtype=np.uint8,
    )
    cam_up      = cv2.resize(cam, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
    heatmap_rgb = (cm.jet(cam_up)[:, :, :3] * 255).astype(np.uint8)
    blended     = cv2.addWeighted(base, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return Image.fromarray(blended)


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def predict(model: nn.Module, pil_image: Image.Image):
    """
    Runs the complete DermaScan AI inference pipeline on a validated image.

    Steps:
    1. Preprocess image (resize, normalise to ImageNet stats).
    2. Register Grad-CAM hooks on model.layer4[-1].
    3. Forward pass → softmax probabilities for all 7 classes.
    4. Back-propagate the predicted class score to capture gradients.
    5. Compute and overlay the Grad-CAM heatmap.
    6. Remove all hooks to keep the cached model clean.

    ``torch.enable_grad()`` is required because Streamlit runs in a
    no-gradient context by default, and Grad-CAM needs a backward pass.

    Parameters
    ----------
    model     : nn.Module  — cached ResNet18 in eval mode.
    pil_image : PIL.Image  — skin-validated input image.

    Returns
    -------
    probs     : np.ndarray  — shape (7,), softmax scores.
    pred_idx  : int         — argmax class index.
    cam_image : PIL.Image   — Grad-CAM overlay.
    """
    tensor   = TRANSFORM(pil_image).unsqueeze(0).to(device)
    grad_cam = GradCAM(model)

    try:
        with torch.enable_grad():
            model.zero_grad()
            logits   = model(tensor)
            probs    = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

            # Scalar backward on the winning class score → populate hooks
            logits[0, pred_idx].backward()

        cam       = grad_cam.compute_cam()
        cam_image = overlay_heatmap(pil_image, cam)

    finally:
        grad_cam.remove_hooks()   # prevent hook accumulation on cached model

    return probs, pred_idx, cam_image


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

# ── Inline SVG Logo ───────────────────────────────────────────────────────
_LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="38" height="38">
  <rect width="40" height="40" rx="9" fill="#2563eb"/>
  <circle cx="19" cy="18" r="8" fill="none" stroke="white" stroke-width="2.2"/>
  <circle cx="19" cy="18" r="3" fill="white"/>
  <line x1="11" y1="18" x2="14.5" y2="18" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
  <line x1="23.5" y1="18" x2="27" y2="18" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
  <line x1="19" y1="10" x2="19" y2="13.5" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
  <line x1="19" y1="22.5" x2="19" y2="26" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
  <line x1="24.5" y1="23.5" x2="29" y2="28" stroke="white" stroke-width="2.4" stroke-linecap="round"/>
</svg>
"""


def render_sidebar():
    """Renders the informational sidebar: logo, model info, usage guide, class legend."""
    with st.sidebar:
        st.markdown(
            f'<div class="sb-brand-row">{_LOGO_SVG}'
            '<div>'
            '<div class="sb-brand-name">DermaScan AI</div>'
            '<div class="sb-tagline">AI-Powered Skin Cancer Detection</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown('<div class="sb-section">Model Info</div>', unsafe_allow_html=True)
        for key, val in [
            ("Architecture", "ResNet18"),
            ("Training",     "HAM10000 dataset"),
            ("XAI Method",   "Grad-CAM"),
            ("Images",       "10,015 scans"),
            ("Classes",      "7 diagnostic"),
        ]:
            st.markdown(
                f'<div class="sb-row"><span class="sb-key">{key}</span>'
                f'<span class="sb-val">{val}</span></div>',
                unsafe_allow_html=True,
            )
        st.divider()

        st.markdown('<div class="sb-section">How to Use</div>', unsafe_allow_html=True)
        for num, step in enumerate([
            "Upload a dermoscopy or skin image",
            "AI validates ≥ 15 % skin-tone pixels",
            "View diagnosis + confidence scores",
            "Explore Grad-CAM heatmap",
        ], 1):
            st.markdown(
                f'<div class="sb-step-row">'
                f'<span class="sb-step-num">{num}</span>'
                f'<span>{step}</span></div>',
                unsafe_allow_html=True,
            )
        st.divider()

        st.markdown('<div class="sb-section">Class Reference</div>', unsafe_allow_html=True)
        _risk_cls = {"Benign": "sb-benign", "Pre-cancerous": "sb-pre", "Malignant": "sb-malignant"}
        for info in CLASS_INFO.values():
            st.markdown(
                f'<div class="sb-class-row">'
                f'<span>{info["icon"]}</span>'
                f'<span class="sb-abbr">{info["abbr"]}</span>'
                f'<span style="flex:1;color:var(--text-second)">{info["name"]}</span>'
                f'<span class="{_risk_cls[info["risk"]]}">{info["risk"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption("DermaScan AI v1.0.0 · 2025–26")


# ═══════════════════════════════════════════════════════════════════════════════
#  Static Content
# ═══════════════════════════════════════════════════════════════════════════════

_DISCLAIMER = (
    '<div class="disclaimer">'
    '<strong>⚠️ Medical Disclaimer:</strong> DermaScan AI is an academic research tool '
    'and does <strong>NOT</strong> constitute a clinical medical diagnosis. '
    'Always consult a licensed dermatologist for evaluation of any skin lesion. '
    'This software is provided solely for educational and research purposes.'
    '</div>'
)

_FOOTER = (
    '<div class="hc-footer">'
    '<strong>DermaScan AI</strong> &nbsp;·&nbsp; '
    'HAM10000 · ResNet18 · Grad-CAM XAI<br>'
    'B.E. Computer Engineering · 2025–26'
    '</div>'
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Entry point for the DermaScan AI Streamlit web application.

    Application flow
    ----------------
    1. Inject CSS and render sidebar.
    2. Display page header and medical disclaimer.
    3. Load (and cache) the trained ResNet18 model.
    4. Accept image upload → validate skin content.
    5. Run inference + Grad-CAM.
    6. Render metrics, confidence bar chart, heatmap, and result card.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
    render_sidebar()

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="page-header">'
        f'{_LOGO_SVG}'
        f'<div class="page-header-text">'
        f'<span class="page-title">Skin Lesion Analysis</span>'
        f'<span class="page-meta">AI-powered dermoscopy analysis &nbsp;&middot;&nbsp; '
        f'ResNet18 + Grad-CAM XAI &nbsp;&middot;&nbsp; HAM10000 &middot; 7 diagnostic classes</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_DISCLAIMER, unsafe_allow_html=True)

    # ── Model Loading ────────────────────────────────────────────────────────
    try:
        model = load_model()
    except FileNotFoundError:
        st.error(
            "\U0001f6ab **Model weights not found** (`model_final.pth` is missing).  \n\n"
            "Please run `python main.py` to train the model and generate the weights file."
        )
        st.markdown(_FOOTER, unsafe_allow_html=True)
        return

    # ── Upload ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hc-label" style="margin-bottom:0.5rem;">Upload Skin Lesion Image</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Supported formats: JPG \u00b7 JPEG \u00b7 PNG",
        type             = ["jpg", "jpeg", "png"],
        label_visibility = "collapsed",
    )

    if uploaded is None:
        st.info(
            "\U0001f446 Upload a clear, close-up dermoscopy or clinical photograph "
            "of a skin lesion to begin the AI analysis."
        )
        st.markdown(_FOOTER, unsafe_allow_html=True)
        return

    pil_image = Image.open(uploaded).convert("RGB")

    # ── Image preview + Skin Validation ─────────────────────────────────────
    col_img, col_val = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="hc-label">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)

    with col_val:
        st.markdown('<div class="hc-label">Image Validation</div>', unsafe_allow_html=True)
        with st.spinner("Checking skin-pixel content\u2026"):
            skin_ok = is_skin_image(pil_image)

        if not skin_ok:
            st.error(
                "\U0001f6ab **Invalid Image \u2014 Please upload a valid skin image.**\n\n"
                "The uploaded file does not appear to contain a skin lesion. "
                "Only clear, close-up photographs or dermoscopy images of skin are accepted."
            )
            st.markdown(_FOOTER, unsafe_allow_html=True)
            return

        w, h = pil_image.size
        st.success("\u2705 Skin content validated \u2014 proceeding to AI analysis.")
        st.markdown(f"**Resolution:** {w} \u00d7 {h} px")

    st.divider()

    # ── Inference ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hc-label" style="margin-bottom:0.5rem;">AI Analysis Results</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("Running DermaScan AI inference\u2026"):
        probs, pred_idx, cam_image = predict(model, pil_image)

    info = CLASS_INFO[pred_idx]

    # ── Top-level Metrics ────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Class", info["name"])
    m2.metric("Confidence",      f"{probs[pred_idx] * 100:.1f}%")
    m3.metric("Risk Level",      f"{info['icon']} {info['risk']}")
    st.divider()

    # ── Confidence Scores + Grad-CAM ─────────────────────────────────────────
    col_scores, col_cam = st.columns([1, 1], gap="large")

    with col_scores:
        st.markdown(
            '<div class="hc-label">Confidence Scores \u2014 All 7 Classes</div>',
            unsafe_allow_html=True,
        )
        for idx in np.argsort(probs)[::-1]:
            c   = CLASS_INFO[idx]
            pct = probs[idx] * 100
            st.markdown(
                f'<div class="conf-row">'
                f'<span class="conf-label">{c["icon"]} <b>{c["abbr"]}</b>'
                f' &nbsp;\u2014&nbsp; {c["name"]}</span>'
                f'<span class="conf-pct">{pct:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(float(probs[idx]))

    with col_cam:
        st.markdown(
            '<div class="hc-label">Grad-CAM Explainability Heatmap</div>',
            unsafe_allow_html=True,
        )
        st.image(cam_image, use_container_width=True)
        st.caption(
            "**Red / yellow** = regions the model focused on most.  \n"
            "**Blue** = regions with little influence on the prediction.  \n"
            "*(Grad-CAM \u2014 Selvaraju et al., 2017)*"
        )

    # ── Detailed Result Card ─────────────────────────────────────────────────
    risk_chip = (
        '<span class="chip chip-red">Malignant</span>'       if info["risk"] == "Malignant"
        else '<span class="chip chip-amber">Pre-cancerous</span>' if info["risk"] == "Pre-cancerous"
        else '<span class="chip chip-green">Benign</span>'
    )
    st.markdown(
        f'<div class="result-card">'
        f'<div style="display:flex;align-items:center;gap:0.75rem;">'
        f'<p class="result-title">{info["icon"]} {info["name"]} ({info["abbr"]})</p>'
        f'{risk_chip}'
        f'</div>'
        f'<p class="result-body">'
        f'This classification was produced by a ResNet18 deep learning model fine-tuned on the '
        f'<strong>HAM10000 dermoscopy dataset</strong> (\u223c10,015 images, 7 diagnostic categories). '
        f'Class imbalance was addressed during training with <em>WeightedRandomSampler</em>. '
        f'The Grad-CAM heatmap visualises the spatial regions that most influenced this prediction.'
        f'</p>'
        f'<p class="result-warn">\u26a0\ufe0f This output is for research and educational purposes only. '
        f'Please consult a licensed dermatologist for any clinical decisions.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(_FOOTER, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

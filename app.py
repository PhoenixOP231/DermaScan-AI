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

_CSS = """
<style>
/* ═══════════════════ Google Fonts ══════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════ CSS Custom Properties ══════════════════════════════════
   Dark theme by default; light theme via prefers-color-scheme               */
:root {
    --bg-primary      : #080c14;
    --bg-secondary    : #0e1521;
    --bg-card         : #111827;
    --bg-glass        : rgba(17, 24, 39, 0.75);
    --border          : rgba(99, 179, 237, 0.12);
    --border-bright   : rgba(99, 179, 237, 0.25);
    --text-primary    : #f0f6ff;
    --text-secondary  : #b8c8e0;
    --text-muted      : #6b8299;
    --text-faint      : #3d5166;
    --accent-1        : #6366f1;
    --accent-2        : #06b6d4;
    --accent-3        : #8b5cf6;
    --gradient-hero   : linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
    --gradient-card   : linear-gradient(145deg, #111827 0%, #0e1521 100%);
    --success         : #10b981;
    --warning         : #f59e0b;
    --danger          : #ef4444;
    --scrollbar-thumb : #1e3a5f;
    --shadow-card     : 0 4px 24px rgba(0,0,0,0.4), 0 1px 4px rgba(99,179,237,0.06);
    --shadow-glow     : 0 0 30px rgba(99, 102, 241, 0.15);
}
@media (prefers-color-scheme: light) {
    :root {
        --bg-primary      : #f0f4f8;
        --bg-secondary    : #ffffff;
        --bg-card         : #ffffff;
        --bg-glass        : rgba(255,255,255,0.85);
        --border          : #dde3ea;
        --border-bright   : #c5d0de;
        --text-primary    : #0f1923;
        --text-secondary  : #2d3d4f;
        --text-muted      : #5a6e82;
        --text-faint      : #8fa3b8;
        --accent-1        : #4f46e5;
        --accent-2        : #0891b2;
        --accent-3        : #7c3aed;
        --gradient-hero   : linear-gradient(135deg, #4f46e5 0%, #0891b2 100%);
        --gradient-card   : linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        --success         : #059669;
        --warning         : #d97706;
        --danger          : #dc2626;
        --scrollbar-thumb : #c5d0de;
        --shadow-card     : 0 2px 16px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.06);
        --shadow-glow     : 0 0 30px rgba(79, 70, 229, 0.08);
    }
}

/* ═══════════════════ Base ═══════════════════════════════════════════════════ */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background    : var(--bg-primary) !important;
    color         : var(--text-secondary) !important;
    font-family   : 'Inter', system-ui, -apple-system, sans-serif !important;
    font-size     : 15px;
    line-height   : 1.6;
}

/* ── Wipe default Streamlit padding ── */
.block-container {
    padding-top   : 1.25rem !important;
    padding-left  : 1.75rem !important;
    padding-right : 1.75rem !important;
    max-width     : 1160px  !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background    : var(--bg-secondary) !important;
    border-right  : 1px solid var(--border);
    padding-top   : 1rem;
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

/* ══════════════════ Hero Banner ════════════════════════════════════════════ */
.hero-banner {
    background    : var(--gradient-hero);
    border-radius : 18px;
    padding       : 2.5rem 2rem 2rem 2rem;
    margin-bottom : 1.5rem;
    position      : relative;
    overflow      : hidden;
    box-shadow    : var(--shadow-glow);
}
.hero-banner::before {
    content       : '';
    position      : absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background    : rgba(255,255,255,0.06);
    border-radius : 50%;
}
.hero-banner::after {
    content       : '';
    position      : absolute;
    bottom: -80px; left: -40px;
    width: 200px; height: 200px;
    background    : rgba(255,255,255,0.04);
    border-radius : 50%;
}
.hero-logo-row {
    display       : flex;
    align-items   : center;
    gap           : 1rem;
    margin-bottom : 0.5rem;
}
.hero-title {
    font-size     : 2.4rem;
    font-weight   : 800;
    color         : #ffffff !important;
    letter-spacing: -0.02em;
    margin        : 0;
    line-height   : 1.15;
}
.hero-subtitle {
    font-size     : 1rem;
    color         : rgba(255,255,255,0.78) !important;
    font-weight   : 400;
    margin-top    : 0.35rem;
    letter-spacing: 0.01em;
}
.hero-badge {
    display       : inline-block;
    background    : rgba(255,255,255,0.15);
    color         : #fff !important;
    border        : 1px solid rgba(255,255,255,0.25);
    border-radius : 20px;
    padding       : 0.2rem 0.8rem;
    font-size     : 0.72rem;
    font-weight   : 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top    : 0.75rem;
}

/* ══════════════════ Cards ══════════════════════════════════════════════════ */
.ds-card {
    background    : var(--gradient-card);
    border        : 1px solid var(--border);
    border-radius : 16px;
    padding       : 1.5rem;
    margin-bottom : 1rem;
    box-shadow    : var(--shadow-card);
    transition    : border-color 0.2s ease, box-shadow 0.2s ease;
}
.ds-card:hover {
    border-color  : var(--border-bright);
    box-shadow    : var(--shadow-card), var(--shadow-glow);
}
.ds-card-title {
    font-size     : 0.7rem;
    font-weight   : 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color         : var(--text-muted) !important;
    margin-bottom : 0.5rem;
}

/* ══════════════════ Result Card ════════════════════════════════════════════ */
.result-card {
    background    : var(--gradient-card);
    border        : 1px solid var(--border);
    border-radius : 16px;
    padding       : 1.5rem 1.75rem;
    margin-top    : 1.5rem;
    box-shadow    : var(--shadow-card);
}
.result-card p { color: var(--text-muted); line-height: 1.75; }
.result-card .warn-text { color: var(--danger); font-weight: 600; }

/* ══════════════════ Metric Boxes ═══════════════════════════════════════════ */
[data-testid="stMetric"] {
    background    : var(--gradient-card) !important;
    border        : 1px solid var(--border) !important;
    border-radius : 14px !important;
    padding       : 1rem 1.25rem !important;
    box-shadow    : var(--shadow-card);
}
[data-testid="stMetricLabel"] * {
    font-family   : 'Inter', sans-serif !important;
    font-size     : 0.72rem !important;
    font-weight   : 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color         : var(--text-muted) !important;
}
[data-testid="stMetricValue"] * {
    font-family   : 'Inter', sans-serif !important;
    font-size     : 1.35rem !important;
    font-weight   : 700 !important;
    color         : var(--text-primary) !important;
}

/* ══════════════════ Progress Bars ══════════════════════════════════════════ */
.stProgress > div > div > div > div {
    background    : linear-gradient(90deg, var(--accent-1), var(--accent-2)) !important;
    border-radius : 99px !important;
}
.stProgress > div > div > div {
    background    : var(--bg-card) !important;
    border-radius : 99px !important;
}

/* ══════════════════ File Uploader ══════════════════════════════════════════ */
[data-testid="stFileUploader"] section {
    border        : 2px dashed var(--border-bright) !important;
    border-radius : 14px !important;
    background    : var(--bg-card) !important;
    min-height    : 90px;
    transition    : border-color 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color  : var(--accent-2) !important;
    background    : var(--bg-secondary) !important;
}

/* ══════════════════ Buttons ════════════════════════════════════════════════ */
.stButton > button {
    background    : var(--gradient-hero);
    color         : #ffffff !important;
    border        : none !important;
    border-radius : 10px;
    padding       : 0.65rem 1.75rem;
    font-family   : 'Inter', sans-serif !important;
    font-weight   : 600;
    font-size     : 0.9rem;
    letter-spacing: 0.02em;
    width         : 100%;
    transition    : opacity 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
    touch-action  : manipulation;
    box-shadow    : 0 2px 12px rgba(99,102,241,0.3);
}
.stButton > button:hover  { opacity: 0.88; box-shadow: 0 4px 20px rgba(99,102,241,0.4); }
.stButton > button:active { transform: scale(0.97); }

/* ══════════════════ Typography ═════════════════════════════════════════════ */
h1, h2, h3, h4, h5 {
    font-family   : 'Inter', sans-serif !important;
    color         : var(--text-primary) !important;
    letter-spacing: -0.01em;
    font-weight   : 700;
}
code, pre, [data-testid="stCode"] * {
    font-family   : 'JetBrains Mono', monospace !important;
    font-size     : 0.85rem !important;
}

/* ══════════════════ Alerts ═════════════════════════════════════════════════ */
[data-testid="stAlert"] { border-radius: 12px !important; }

/* ══════════════════ Image containers ══════════════════════════════════════ */
[data-testid="stImage"] img {
    border-radius : 12px;
    max-width     : 100%;
    height        : auto;
    box-shadow    : var(--shadow-card);
}

/* ══════════════════ Divider ════════════════════════════════════════════════ */
hr { border-color: var(--border) !important; }

/* ══════════════════ Caption / small text ═══════════════════════════════════ */
.stCaption, small, caption {
    font-family   : 'Inter', sans-serif !important;
    color         : var(--text-muted) !important;
    font-size     : 0.78rem !important;
}

/* ══════════════════ Scrollbar ══════════════════════════════════════════════ */
::-webkit-scrollbar             { width: 5px; height: 5px; }
::-webkit-scrollbar-track       { background: var(--bg-primary); }
::-webkit-scrollbar-thumb       { background: var(--scrollbar-thumb); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ══════════════════ Footer ═════════════════════════════════════════════════ */
.footer {
    margin-top    : 3rem;
    padding       : 1.75rem 1rem;
    text-align    : center;
    background    : var(--bg-card);
    border-top    : 1px solid var(--border);
    border-radius : 16px;
    color         : var(--text-faint);
    font-family   : 'Inter', sans-serif;
    font-size     : 0.8rem;
    line-height   : 2;
}
.footer strong { color: var(--text-primary) !important; font-size: 0.95rem; }
.footer .tag   { color: var(--text-muted); font-style: italic; }
.footer .dot   { color: var(--accent-2); margin: 0 0.35rem; }

/* ══════════════════ Sidebar logo strip ═════════════════════════════════════ */
.sidebar-logo {
    display       : flex;
    align-items   : center;
    gap           : 0.6rem;
    margin-bottom : 0.25rem;
}
.sidebar-logo-text {
    font-family   : 'Inter', sans-serif;
    font-size     : 1.1rem;
    font-weight   : 800;
    letter-spacing: -0.02em;
    background    : var(--gradient-hero);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ══════════════════ Confidence row ═════════════════════════════════════════ */
.conf-row {
    display        : flex;
    align-items    : center;
    justify-content: space-between;
    margin-bottom  : 0.15rem;
    font-size      : 0.82rem;
    font-family    : 'Inter', sans-serif;
    color          : var(--text-secondary);
}
.conf-pct {
    font-family    : 'JetBrains Mono', monospace;
    font-size      : 0.78rem;
    color          : var(--accent-2);
    font-weight    : 500;
}

/* ══════════════════ Mobile — ≤ 768 px ═══════════════════════════════════════ */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction : column !important;
        gap            : 0.5rem;
    }
    [data-testid="column"] {
        width     : 100% !important;
        flex      : 1 1 100% !important;
        min-width : 0 !important;
    }
    .hero-title    { font-size: 1.6rem  !important; }
    .hero-subtitle { font-size: 0.88rem !important; }
    .block-container {
        padding-left  : 0.75rem !important;
        padding-right : 0.75rem !important;
    }
    [data-testid="stFileUploader"] section { min-height: 110px; }
    .footer { font-size: 0.72rem; padding: 1rem; }
}

/* ══════════════════ Mobile — ≤ 480 px ═══════════════════════════════════════ */
@media (max-width: 480px) {
    .hero-banner  { padding: 1.5rem 1.25rem; }
    .hero-title   { font-size: 1.35rem !important; }
    .block-container {
        padding-left  : 0.5rem !important;
        padding-right : 0.5rem !important;
    }
    .result-card    { padding: 1rem; }
    .ds-card        { padding: 1rem; }
}
</style>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH  = "model_final.pth"
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
    state_dict = torch.load(MODEL_PATH, map_location=device)
    state_dict = {k: v.float() for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Security Gatekeeper — Skin-Pixel Validation
# ═══════════════════════════════════════════════════════════════════════════════

def is_skin_image(pil_image: Image.Image, threshold: float = 0.15) -> bool:
    """
    Verifies that the uploaded image contains a sufficient proportion of
    skin-tone pixels before allowing model inference.

    Applies the Kovac et al. (2002) explicit RGB skin segmentation rules:

        R > 95  AND  G > 40  AND  B > 20
        max(R,G,B) − min(R,G,B) > 15
        |R − G| > 15  AND  R > G  AND  R > B

    These rules have high specificity across a wide range of skin tones
    under standard illumination.  If fewer than ``threshold`` (default 15%)
    of all pixels satisfy every rule, the image is rejected to prevent
    the model from producing misleading predictions on non-dermatological
    photographs.

    Parameters
    ----------
    pil_image : PIL.Image
        The user-uploaded image (any size, any mode).
    threshold : float
        Minimum fraction of pixels (0–1) that must register as skin.

    Returns
    -------
    bool
        True  — image passes the skin-content test.
        False — rejection; caller should display an error to the user.
    """
    img = np.array(pil_image.convert("RGB"), dtype=np.float32)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    skin_mask = (
          (R > 95) & (G > 40) & (B > 20)
        & (  np.maximum(np.maximum(R, G), B)
           - np.minimum(np.minimum(R, G), B) > 15)
        & (np.abs(R - G) > 15)
        & (R > G) & (R > B)
    )

    skin_ratio = float(skin_mask.sum()) / float(skin_mask.size)
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
_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="42" height="42">
  <defs>
    <linearGradient id="lg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#6366f1"/>
      <stop offset="100%" stop-color="#06b6d4"/>
    </linearGradient>
  </defs>
  <!-- Outer circle -->
  <circle cx="24" cy="24" r="23" fill="url(#lg)" opacity="0.15" stroke="url(#lg)" stroke-width="1.5"/>
  <!-- Microscope body -->
  <rect x="20" y="8" width="8" height="14" rx="2" fill="url(#lg)"/>
  <!-- Lens -->
  <ellipse cx="24" cy="23" rx="6" ry="4" fill="none" stroke="url(#lg)" stroke-width="2"/>
  <!-- Stand arm -->
  <path d="M24 27 L24 34" stroke="url(#lg)" stroke-width="2.5" stroke-linecap="round"/>
  <!-- Base -->
  <path d="M14 38 Q24 34 34 38" stroke="url(#lg)" stroke-width="2.5" stroke-linecap="round" fill="none"/>
  <!-- Scan lines (AI indicator) -->
  <line x1="30" y1="18" x2="36" y2="18" stroke="#06b6d4" stroke-width="1.5" stroke-linecap="round" opacity="0.8"/>
  <line x1="30" y1="22" x2="38" y2="22" stroke="#6366f1" stroke-width="1.5" stroke-linecap="round" opacity="0.6"/>
  <line x1="30" y1="26" x2="35" y2="26" stroke="#06b6d4" stroke-width="1.5" stroke-linecap="round" opacity="0.4"/>
</svg>
"""


def render_sidebar():
    """Renders the informational sidebar: logo, project info, usage guide, class legend."""
    with st.sidebar:
        # Logo + brand name
        st.markdown(
            f"""
            <div class="sidebar-logo">
                {_LOGO_SVG}
                <span class="sidebar-logo-text">DermaScan AI</span>
            </div>
            <div style="font-size:0.72rem; color:var(--text-muted); margin-bottom:0.75rem;
                        font-family:'Inter',sans-serif; letter-spacing:0.04em;">
                AI-Powered Skin Cancer Detection
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown(
            "<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.12em;"
            "text-transform:uppercase;color:var(--text-muted);font-family:Inter,sans-serif;'"
            ">Model Info</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "**Architecture:** ResNet18  \n"
            "**Training:** HAM10000 dataset  \n"
            "**XAI:** Grad-CAM (Selvaraju 2017)  \n"
            "**Images:** 10,015 dermoscopy scans  \n"
            "**Classes:** 7 diagnostic categories"
        )
        st.divider()

        st.markdown(
            "<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.12em;"
            "text-transform:uppercase;color:var(--text-muted);font-family:Inter,sans-serif;'"
            ">How to Use</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "**1 ·** Upload a dermoscopy or skin image  \n"
            "**2 ·** AI validates ≥ 15% skin-tone pixels  \n"
            "**3 ·** View diagnosis + confidence scores  \n"
            "**4 ·** Explore Grad-CAM heatmap"
        )
        st.divider()

        st.markdown(
            "<p style='font-size:0.7rem;font-weight:700;letter-spacing:0.12em;"
            "text-transform:uppercase;color:var(--text-muted);font-family:Inter,sans-serif;'"
            ">Class Reference</p>",
            unsafe_allow_html=True,
        )
        for info in CLASS_INFO.values():
            st.markdown(
                f"<span style='font-size:0.83rem;font-family:Inter,sans-serif;'>"
                f"{info['icon']} <b>{info['abbr']}</b> &nbsp;—&nbsp; {info['name']}</span>",
                unsafe_allow_html=True,
            )
        st.divider()
        st.markdown("🟢 Benign &nbsp;·&nbsp; 🟡 Pre-cancerous &nbsp;·&nbsp; 🔴 Malignant")
        st.divider()
        st.caption("DermaScan AI v1.0.0 · 2025–26")


# ═══════════════════════════════════════════════════════════════════════════════
#  Static Content
# ═══════════════════════════════════════════════════════════════════════════════

_DISCLAIMER = (
    "⚠️ **Medical Disclaimer:** DermaScan AI is an academic research tool and does "
    "**NOT** constitute a clinical medical diagnosis. Always consult a licensed "
    "dermatologist for evaluation of any skin lesion. This software is provided "
    "solely for educational and research purposes."
)

_FOOTER = """
<div class="footer">
    <strong>DermaScan AI</strong>
    <span class="dot">◆</span>
    <span class="tag">HAM10000 · ResNet18 · Grad-CAM XAI</span><br>
    <span style="font-size:0.76rem; font-family:'Inter',sans-serif;">
        3rd Year Engineering Project
        <span class="dot">·</span>
        B.E. Computer Engineering
        <span class="dot">·</span>
        Academic Year 2025–26
    </span>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Entry point for the DermaScan AI Streamlit web application.

    Application flow
    ----------------
    1. Inject CSS and render sidebar.
    2. Display header and medical disclaimer.
    3. Load (and cache) the trained ResNet18 model.
    4. Accept image upload → validate skin content.
    5. Run inference + Grad-CAM.
    6. Render metrics, confidence bar chart, heatmap, and result card.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
    render_sidebar()

    # ── Hero Banner ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-logo-row">
                {_LOGO_SVG}
                <div>
                    <div class="hero-title">DermaScan AI</div>
                </div>
            </div>
            <div class="hero-subtitle">
                AI-Powered Skin Cancer Detection &nbsp;·&nbsp;
                ResNet18 + Grad-CAM Explainability
            </div>
            <div class="hero-badge">HAM10000 · 7 Classes · FP16 Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(_DISCLAIMER)
    st.divider()

    # ── Model Loading ────────────────────────────────────────────────────────
    try:
        model = load_model()
    except FileNotFoundError:
        st.error(
            "🚫 **Model weights not found** (`model_final.pth` is missing).  \n\n"
            "Please run `python main.py` to train the model and generate the weights file."
        )
        st.markdown(_FOOTER, unsafe_allow_html=True)
        return

    # ── Upload ───────────────────────────────────────────────────────────────
    st.markdown("## 📤 Upload Skin Lesion Image")
    uploaded = st.file_uploader(
        "Supported formats: JPG · JPEG · PNG",
        type             = ["jpg", "jpeg", "png"],
        label_visibility = "collapsed",
    )

    if uploaded is None:
        st.info(
            "👆 Upload a clear, close-up dermoscopy or clinical photograph "
            "of a skin lesion to begin the AI analysis."
        )
        st.markdown(_FOOTER, unsafe_allow_html=True)
        return

    pil_image = Image.open(uploaded).convert("RGB")

    # ── Image preview + Skin Validation ─────────────────────────────────────
    col_img, col_val = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown(
            "<div class='ds-card-title' style='font-family:Inter,sans-serif;'>Uploaded Image</div>",
            unsafe_allow_html=True,
        )
        st.image(pil_image, use_container_width=True)

    with col_val:
        st.markdown(
            "<div class='ds-card-title' style='font-family:Inter,sans-serif;'>🛡️ Image Validation</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Checking skin-pixel content…"):
            skin_ok = is_skin_image(pil_image)

        if not skin_ok:
            st.error(
                "**🚫 Skin Validation Failed**\n\n"
                "The uploaded image contains **< 15% skin-tone pixels**.  \n"
                "Please upload a clear, close-up photograph of a skin lesion "
                "(dermoscopy image or clinical skin photo)."
            )
            st.markdown(_FOOTER, unsafe_allow_html=True)
            return

        w, h = pil_image.size
        st.success("✅ Skin content validated — proceeding to AI analysis.")
        st.markdown(f"**Resolution:** {w} × {h} px")

    st.divider()

    # ── Inference ────────────────────────────────────────────────────────────
    st.markdown("## 🧠 AI Analysis")
    with st.spinner("Running DermaScan AI inference…"):
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
            "<div class='ds-card-title' style='font-family:Inter,sans-serif;'"
            ">📊 Confidence Scores — All 7 Classes</div>",
            unsafe_allow_html=True,
        )
        for idx in np.argsort(probs)[::-1]:
            c   = CLASS_INFO[idx]
            pct = probs[idx] * 100
            st.markdown(
                f"<div class='conf-row'>"
                f"<span>{c['icon']} <b>{c['abbr']}</b> &nbsp;—&nbsp; {c['name']}</span>"
                f"<span class='conf-pct'>{pct:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(probs[idx]))

    with col_cam:
        st.markdown(
            "<div class='ds-card-title' style='font-family:Inter,sans-serif;'"
            ">🌡️ Grad-CAM Explainability Heatmap</div>",
            unsafe_allow_html=True,
        )
        st.image(cam_image, use_container_width=True)
        st.caption(
            "**Red / yellow** = regions the model focused on most.  \n"
            "**Blue** = regions with little influence on the prediction.  \n"
            "*(Grad-CAM — Selvaraju et al., 2017)*"
        )

    # ── Detailed Result Card ─────────────────────────────────────────────────
    risk_colour = (
        "#e63946" if info["risk"] == "Malignant"
        else "#f4a261" if info["risk"] == "Pre-cancerous"
        else "#3fb950"
    )
    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="color:{risk_colour}; margin-bottom:0.4rem;">
                {info['icon']} {info['risk']} &nbsp;·&nbsp;
                {info['name']} ({info['abbr']})
            </h3>
            <p>
                This classification was produced by a ResNet18 deep learning model
                fine-tuned on the <strong>HAM10000 dermoscopy dataset</strong>
                (~10,015 images, 7 diagnostic categories).
                Class imbalance was addressed during training with
                <em>WeightedRandomSampler</em>. The Grad-CAM heatmap above
                visualises the spatial regions that most influenced this prediction.
            </p>
            <p class="warn-text">
                ⚠️ This output is for research and educational purposes only.
                Please consult a licensed dermatologist for any clinical decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(_FOOTER, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

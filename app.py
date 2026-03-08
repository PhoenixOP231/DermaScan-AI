"""
app.py — DermaScan AI Streamlit Web Application
================================================
AI-powered skin cancer detection with:
  • Skin-pixel gatekeeper (Kovac et al. RGB rules)
  • Grad-CAM explainability heatmap overlay
  • Auto dark / light mode via CSS custom properties
  • Fully mobile-responsive (Android · iOS · tablet · desktop)

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
/* ═══════════════════ CSS Custom Properties ══════════════════════════════════
   Automatically switches between dark and light themes based on the OS /
   browser preference via prefers-color-scheme.                              */
:root {
    --bg-primary     : #0d1117;
    --bg-secondary   : #161b22;
    --bg-tertiary    : #21262d;
    --border         : #30363d;
    --text-primary   : #e6edf3;
    --text-secondary : #c9d1d9;
    --text-muted     : #8b949e;
    --text-faint     : #6e7681;
    --accent-start   : #6e40c9;
    --accent-end     : #2ea8e2;
    --success        : #3fb950;
    --warning        : #f4a261;
    --danger         : #e63946;
    --heart          : #e63946;
    --scrollbar-thumb: #30363d;
}
@media (prefers-color-scheme: light) {
    :root {
        --bg-primary     : #f6f8fa;
        --bg-secondary   : #ffffff;
        --bg-tertiary    : #eaeef2;
        --border         : #d0d7de;
        --text-primary   : #1f2328;
        --text-secondary : #24292f;
        --text-muted     : #57606a;
        --text-faint     : #848d97;
        --accent-start   : #5a32a3;
        --accent-end     : #1a7fb5;
        --success        : #1a7f37;
        --warning        : #b35900;
        --danger         : #cf222e;
        --heart          : #cf222e;
        --scrollbar-thumb: #d0d7de;
    }
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color : var(--bg-primary) !important;
    color            : var(--text-secondary) !important;
    font-family      : 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont,
                       'Helvetica Neue', Arial, sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background   : var(--bg-secondary) !important;
    border-right : 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
}

/* ── Main block container ── */
.block-container {
    padding-top   : 1.5rem  !important;
    padding-left  : 1.5rem  !important;
    padding-right : 1.5rem  !important;
    max-width     : 1100px  !important;
}

/* ── File uploader zone ── */
[data-testid="stFileUploader"] section {
    border        : 2px dashed var(--border) !important;
    border-radius : 12px;
    background    : var(--bg-secondary) !important;
    min-height    : 80px;
    cursor        : pointer;
}

/* ── Primary buttons ── */
.stButton > button {
    background    : linear-gradient(135deg, var(--accent-start) 0%,
                                            var(--accent-end)   100%);
    color         : #ffffff !important;
    border        : none;
    border-radius : 8px;
    padding       : 0.6rem 1.5rem;
    font-weight   : 600;
    font-size     : 1rem;
    width         : 100%;
    transition    : opacity 0.2s ease, transform 0.1s ease;
    touch-action  : manipulation;   /* remove 300 ms tap delay on mobile */
}
.stButton > button:hover  { opacity: 0.85; }
.stButton > button:active { transform: scale(0.97); }

/* ── Progress bars ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-start),
                                       var(--accent-end)) !important;
}

/* ── Metric boxes ── */
[data-testid="stMetric"] {
    background    : var(--bg-secondary) !important;
    border        : 1px solid var(--border);
    border-radius : 10px;
    padding       : 0.75rem 1rem;
    margin-bottom : 0.5rem;
}
[data-testid="stMetricLabel"] * { color: var(--text-muted)    !important; }
[data-testid="stMetricValue"] * { color: var(--text-primary)  !important; }

/* ── Headings ── */
h1, h2, h3, h4 { color: var(--text-primary) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Alert / info / warning boxes ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Image containers ── */
[data-testid="stImage"] img {
    border-radius : 10px;
    max-width     : 100%;
    height        : auto;
}

/* ── Result card ── */
.result-card {
    background    : var(--bg-secondary);
    border        : 1px solid var(--border);
    border-radius : 14px;
    padding       : 1.25rem 1.5rem;
    margin-top    : 1.5rem;
}
.result-card p {
    color       : var(--text-muted);
    line-height : 1.7;
}
.result-card .warn-text {
    color       : var(--danger);
    font-weight : 600;
}

/* ── Footer ── */
.footer {
    margin-top    : 3rem;
    padding       : 1.5rem 1rem;
    text-align    : center;
    background    : var(--bg-secondary);
    border-top    : 1px solid var(--border);
    border-radius : 12px;
    color         : var(--text-faint);
    font-size     : 0.82rem;
    line-height   : 2;
}
.footer strong { color: var(--text-primary) !important; font-size: 1rem; }
.footer .heart { color: var(--heart); }
.footer .tag   { color: var(--text-muted); font-style: italic; }

/* ═══════════════════════ Scrollbar ════════════════════════════════════════ */
::-webkit-scrollbar             { width: 6px; height: 6px; }
::-webkit-scrollbar-track       { background: var(--bg-primary); }
::-webkit-scrollbar-thumb       { background: var(--scrollbar-thumb);
                                   border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

/* ═══════════════════════ Mobile — ≤ 768 px ════════════════════════════════ */
@media (max-width: 768px) {

    /* Stack all Streamlit columns vertically on narrow screens */
    [data-testid="stHorizontalBlock"] {
        flex-direction : column !important;
        gap            : 0.5rem;
    }
    [data-testid="column"] {
        width     : 100% !important;
        flex      : 1 1 100% !important;
        min-width : 0 !important;
    }

    /* Responsive font sizes */
    h1 { font-size: 1.55rem !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.05rem !important; }
    h4 { font-size: 0.95rem !important; }

    /* Tighten padding on small screens */
    .block-container {
        padding-left  : 0.75rem !important;
        padding-right : 0.75rem !important;
        padding-top   : 0.75rem !important;
    }

    /* Larger touch target for file uploader */
    [data-testid="stFileUploader"] section {
        min-height : 110px;
    }

    /* Metric cards — full width stacking */
    [data-testid="stMetric"] { width: 100%; }

    /* Footer smaller text on phones */
    .footer { font-size: 0.72rem; padding: 1rem 0.75rem; }
}

/* ═══════════════════════ Mobile — ≤ 480 px ════════════════════════════════ */
@media (max-width: 480px) {
    h1 { font-size: 1.3rem !important; }
    .block-container {
        padding-left  : 0.5rem !important;
        padding-right : 0.5rem !important;
    }
    [data-testid="stFileUploader"] section { min-height: 90px; }
    .result-card { padding: 1rem; }
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

def render_sidebar():
    """Renders the informational sidebar: project info, usage guide, class legend."""
    with st.sidebar:
        st.markdown("## 🔬 DermaScan AI")
        st.markdown(
            "**Model:** ResNet18 (ImageNet → HAM10000)  \n"
            "**XAI:** Grad-CAM  \n"
            "**Dataset:** 10,015 dermoscopy images  \n"
            "**Classes:** 7 diagnostic categories"
        )
        st.divider()

        st.markdown("### 📋 How to Use")
        st.markdown(
            "**1️⃣** Upload a dermoscopy or clinical skin image  \n"
            "**2️⃣** AI validates ≥ 15% skin-tone pixel content  \n"
            "**3️⃣** View diagnosis + confidence scores  \n"
            "**4️⃣** Explore the Grad-CAM heatmap for transparency"
        )
        st.divider()

        st.markdown("### 🏷️ Class Reference")
        for info in CLASS_INFO.values():
            st.markdown(f"{info['icon']} **{info['abbr']}** — {info['name']}")
        st.divider()

        st.markdown("### Risk Legend")
        st.markdown("🟢 Benign &nbsp;•&nbsp; 🟡 Pre-cancerous &nbsp;•&nbsp; 🔴 Malignant")
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
    <strong>DermaScan AI</strong><br>
    <span class="tag">AI-Powered Skin Lesion Analysis · HAM10000 · ResNet18 + Grad-CAM</span><br>
    <span style="font-size:0.78rem; color:var(--text-faint);">
        3rd Year Engineering Project &nbsp;·&nbsp; B.E. Computer Engineering
        &nbsp;·&nbsp; Academic Year 2025–26
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

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("# 🔬 DermaScan AI")
    st.markdown("#### AI-Powered Skin Cancer Detection · ResNet18 + Grad-CAM Explainability")
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
        st.markdown("#### Uploaded Image")
        st.image(pil_image, use_container_width=True)

    with col_val:
        st.markdown("#### 🛡️ Image Validation")
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
    m1.metric("🏷️ Predicted Class", info["name"])
    m2.metric("📊 Confidence",       f"{probs[pred_idx] * 100:.1f}%")
    m3.metric("⚠️ Risk Level",       f"{info['icon']} {info['risk']}")
    st.divider()

    # ── Confidence Scores + Grad-CAM ─────────────────────────────────────────
    col_scores, col_cam = st.columns([1, 1], gap="large")

    with col_scores:
        st.markdown("#### 📊 Confidence Scores — All 7 Classes")
        for idx in np.argsort(probs)[::-1]:
            c   = CLASS_INFO[idx]
            pct = probs[idx] * 100
            st.markdown(f"{c['icon']} **{c['abbr']}** — {c['name']}: `{pct:.1f}%`")
            st.progress(float(probs[idx]))

    with col_cam:
        st.markdown("#### 🌡️ Grad-CAM Explainability Heatmap")
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

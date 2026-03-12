"""
export_to_onnx.py — DermaScan AI: PyTorch → ONNX conversion
=============================================================
Run this ONCE locally before your first Vercel deployment.

Inputs:
    backend/model_final.pth        FP16 checkpoint (trained ResNet18)

Outputs:
    frontend/api/model_final.onnx  ONNX model with two outputs:
                                     • logits        (1, 7)
                                     • layer4_acts   (1, 512, 7, 7)
    frontend/api/fc_weights.npy    FC-layer weight matrix (7, 512)
                                   used by the serverless function to
                                   compute Class Activation Maps without
                                   gradient backprop.

Usage:
    # Activate your venv first, then:
    python export_to_onnx.py

Requirements (already in your venv):
    torch, torchvision, numpy

3rd-Year Engineering Project — B.E. Computer Engineering | 2025–26
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
SRC_PTH    = ROOT / "backend" / "model_final.pth"
OUT_DIR    = ROOT / "frontend" / "api"
OUT_ONNX   = OUT_DIR / "model_final.onnx"
OUT_FC_NPY = OUT_DIR / "fc_weights.npy"

IMG_SIZE    = 224
NUM_CLASSES = 7


# ── Wrapper model that exposes layer4 activations as a second output ──────────

class _ResNet18WithActivations(nn.Module):
    """
    Minimal ResNet18 wrapper that returns (logits, layer4_activations).

    The layer4 feature maps are needed by the serverless function to compute
    Class Activation Maps (CAM) without requiring gradient computation.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool
        self.fc      = base.fc   # nn.Sequential(Dropout(0.4), Linear(512, 7))

    def forward(self, x: torch.Tensor):                   # type: ignore[override]
        x            = self.conv1(x)
        x            = self.bn1(x)
        x            = self.relu(x)
        x            = self.maxpool(x)
        x            = self.layer1(x)
        x            = self.layer2(x)
        x            = self.layer3(x)
        layer4_acts  = self.layer4(x)          # ← (1, 512, 7, 7) saved here
        x            = self.avgpool(layer4_acts)
        x            = torch.flatten(x, 1)
        logits       = self.fc(x)              # (1, 7)
        return logits, layer4_acts


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {SRC_PTH}")
    if not SRC_PTH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {SRC_PTH}.\n"
            "Ensure backend/model_final.pth exists before running this script."
        )

    # 1. Build base architecture
    base = models.resnet18(weights=None)
    base.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(base.fc.in_features, NUM_CLASSES),
    )

    # 2. Load FP16 checkpoint → upcast to FP32 (required for CPU inference)
    state = torch.load(SRC_PTH, map_location="cpu")
    state = {k: v.float() for k, v in state.items()}
    base.load_state_dict(state)
    base.eval()

    # 3. Save FC-layer weight matrix for CAM computation
    #    model.fc = Sequential(Dropout, Linear) → fc[1].weight is (7, 512)
    fc_weights = base.fc[1].weight.detach().cpu().numpy()
    np.save(str(OUT_FC_NPY), fc_weights)
    print(f"✓ fc_weights.npy saved  → {OUT_FC_NPY}   shape={fc_weights.shape}")

    # 4. Wrap model for dual output
    dual_model = _ResNet18WithActivations(base)
    dual_model.eval()

    # 5. Export to ONNX (opset 14 for broad compatibility)
    dummy_input = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        torch.onnx.export(
            dual_model,
            dummy_input,
            str(OUT_ONNX),
            opset_version=14,
            input_names=["image"],
            output_names=["logits", "layer4_acts"],
            dynamic_axes={"image": {0: "batch"}},
            do_constant_folding=True,
            verbose=False,
        )

    size_mb = OUT_ONNX.stat().st_size / 1_000_000
    print(f"✓ model_final.onnx saved → {OUT_ONNX}   ({size_mb:.1f} MB)")
    print()
    print("Next steps:")
    print("  1. git add frontend/api/model_final.onnx frontend/api/fc_weights.npy")
    print("  2. git commit -m 'Add ONNX model for Vercel serverless deployment'")
    print("  3. vercel --cwd frontend   (or push to GitHub and import in Vercel dashboard)")


if __name__ == "__main__":
    main()

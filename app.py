"""
app.py — Solar Panel Fault Detection Web App
============================================
Loads the stacking ensemble (3 base models + Logistic Regression meta-learner)
and serves a Flask web interface for single or batch image prediction.

Run:
    pip install flask torch torchvision tensorflow scikit-learn pillow numpy
    python app.py

Then open: http://127.0.0.1:5000
"""

import os
import io
import json
import time
import glob
import pickle
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision import transforms, models as tv_models

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess

# ── Sklearn ───────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CLASSES     = ['Bird-drop', 'Clean', 'Dusty',
               'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class descriptions shown in UI
CLASS_INFO = {
    'Bird-drop'          : 'Bird droppings detected - reducing panel efficiency.',
    'Clean'              : 'Panel is clean and operating at full efficiency.',
    'Dusty'              : 'Dust accumulation detected - causing gradual power loss.',
    'Electrical-damage'  : 'Electrical fault detected - immediate inspection required.',
    'Physical-Damage'    : 'Physical damage detected on panel surface.',
    'Snow-Covered'       : 'Snow coverage detected - blocking solar absorption.',
}

CLASS_SEVERITY = {
    'Bird-drop'          : 'medium',
    'Clean'              : 'ok',
    'Dusty'              : 'low',
    'Electrical-damage'  : 'critical',
    'Physical-Damage'    : 'critical',
    'Snow-Covered'       : 'medium',
}

CLASS_ACTION = {
    'Bird-drop'          : ('cleaning', 'Cleaning Required',  'Schedule a wash to remove bird droppings from the panel surface.'),
    'Clean'              : ('ok',       'No Action Needed',   'Panel is operating normally. Continue routine monitoring.'),
    'Dusty'              : ('cleaning', 'Cleaning Required',  'Clean the panel surface to restore optimal light absorption.'),
    'Electrical-damage'  : ('repair',   'Repair Required',    'Contact a certified technician immediately. Do not operate until inspected.'),
    'Physical-Damage'    : ('repair',   'Repair Required',    'Panel surface is damaged. Arrange inspection and replacement if necessary.'),
    'Snow-Covered'       : ('cleaning', 'Cleaning Required',  'Remove snow from the panel surface to restore power generation.'),
}

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────
class VGG16Detector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        backbone        = tv_models.vgg16(weights=None)
        self.features   = backbone.features
        self.pool       = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.6),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_efficientnet_b3(num_classes=NUM_CLASSES, dropout=0.4):
    enet = tv_models.efficientnet_b3(weights=None)
    in_f = enet.classifier[1].in_features
    enet.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return enet


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-DISCOVER CHECKPOINTS
# ─────────────────────────────────────────────────────────────────────────────
def find_file(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

VGG16_CKPT = find_file([
    "models/vgg16.h5", "model/vgg16.h5", "vgg16.h5",
    *glob.glob("**/vgg16.h5", recursive=True),
])
ENET_CKPT = find_file([
    "models/efficientnet_b3.h5", "model/efficientnet_b3.h5",
    *glob.glob("**/efficientnet_b3.h5", recursive=True),
])
MOB_CKPT = find_file([
    "models/mobilenet.h5", "model/mobilenet.h5",
    *glob.glob("**/mobilenet.h5", recursive=True),
])
META_CKPT = find_file([
    "models/meta_learner.pkl", "model/meta_learner.pkl", "meta_learner.pkl",
])

print(f"  VGG16          : {VGG16_CKPT or '*** NOT FOUND ***'}")
print(f"  EfficientNetB3 : {ENET_CKPT  or '*** NOT FOUND ***'}")
print(f"  MobileNetV2    : {MOB_CKPT   or '*** NOT FOUND ***'}")
print(f"  Meta-learner   : {META_CKPT  or '(not found — will use EfficientNetB3 alone)'}")


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading models …")

vgg_model = VGG16Detector().to(DEVICE)
vgg_ckpt = torch.load(VGG16_CKPT, map_location=DEVICE, weights_only=False)
vgg_model.load_state_dict(
    vgg_ckpt["model_state"] if isinstance(vgg_ckpt, dict) and "model_state" in vgg_ckpt
    else vgg_ckpt
)
vgg_model.eval()
print("  ✔ VGG16")

enet_model = build_efficientnet_b3().to(DEVICE)
enet_model.load_state_dict(
    torch.load(ENET_CKPT, map_location=DEVICE, weights_only=False))
enet_model.eval()
print("  ✔ EfficientNetB3")

mob_model = tf.keras.models.load_model(MOB_CKPT)
print("  ✔ MobileNetV2")

# Meta-learner (Logistic Regression + scaler)
USE_ENSEMBLE = False
meta_clf     = None
meta_scaler  = None

if META_CKPT:
    with open(META_CKPT, "rb") as f:
        meta_bundle  = pickle.load(f)
    meta_clf     = meta_bundle["clf"]
    meta_scaler  = meta_bundle["scaler"]
    USE_ENSEMBLE = True
    print("  ✔ Meta-learner (Logistic Regression ensemble)")
else:
    print("  ⚠  No meta-learner found. Save it first:\n"
          "     import pickle\n"
          "     pickle.dump({'clf': clf, 'scaler': scaler}, open('models/meta_learner.pkl','wb'))")


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
PT_TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE + 32),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_single(pil_img: Image.Image) -> dict:
    """Run all models on one PIL image, return full prediction dict."""
    t0 = time.perf_counter()

    # ── All three models run (for comparative display) ──────────────────────
    inp = PT_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    vgg_prob  = torch.softmax(vgg_model(inp),  dim=1).squeeze().cpu().numpy()
    enet_prob = torch.softmax(enet_model(inp), dim=1).squeeze().cpu().numpy()

    arr      = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    arr      = mob_preprocess(arr)[np.newaxis, ...]
    mob_prob = mob_model(arr, training=False).numpy().squeeze()

    # ── Primary prediction: EfficientNetB3 only ──────────────────────────────
    # EfficientNetB3 is the winner model — used for the final result.
    # VGG16 and MobileNetV2 run in background and are shown as comparative only.
    final_prob = enet_prob
    mode       = "EfficientNetB3"

    idx        = int(np.argmax(final_prob))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "predicted_class" : CLASSES[idx],
        "confidence"      : round(float(final_prob[idx]) * 100, 2),
        "severity"        : CLASS_SEVERITY[CLASSES[idx]],
        "description"     : CLASS_INFO[CLASSES[idx]],
        "mode"            : mode,
        "elapsed_ms"      : round(elapsed_ms, 1),
        "probabilities"   : {
            c: round(float(p) * 100, 2)
            for c, p in zip(CLASSES, final_prob)
        },
        "action_type"     : CLASS_ACTION[CLASSES[idx]][0],
        "action_title"    : CLASS_ACTION[CLASSES[idx]][1],
        "action_detail"   : CLASS_ACTION[CLASSES[idx]][2],
        "model_probs": {
            "VGG16"      : {c: round(float(p)*100,2) for c,p in zip(CLASSES, vgg_prob)},
            "MobileNetV2": {c: round(float(p)*100,2) for c,p in zip(CLASSES, mob_prob)},
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024   # 32 MB max upload

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def allowed(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html",
                           classes=CLASSES,
                           use_ensemble=USE_ENSEMBLE)


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No images uploaded"}), 400

    results = []
    for f in files:
        if not allowed(f.filename):
            results.append({"filename": f.filename,
                             "error": "Unsupported file type"})
            continue
        try:
            img_bytes = f.read()
            pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Thumbnail for display (base64)
            thumb = pil_img.copy()
            thumb.thumbnail((320, 320))
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()

            pred = predict_single(pil_img)
            pred["filename"]  = f.filename
            pred["thumbnail"] = f"data:image/jpeg;base64,{b64}"
            results.append(pred)
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return jsonify(results)


if __name__ == "__main__":
    print(f"\n  Mode : {'Ensemble (Logistic Regression)' if USE_ENSEMBLE else 'Soft Vote fallback'}")
    print(f"  Open : http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
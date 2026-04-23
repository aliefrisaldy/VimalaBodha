import io
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── MODEL_PATH: pakai path relatif agar jalan di Railway ──────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pth"

DEFAULT_CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]

IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        class_names = checkpoint["class_names"]
        print(f"✅  Class names dari checkpoint: {class_names}")
    else:
        class_names = DEFAULT_CLASS_NAMES
        print(f"⚠️  class_names tidak ada di checkpoint, pakai default.")

    num_classes = len(class_names)

    net = models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    net = net.to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    net.load_state_dict(state_dict)
    net.eval()
    return net, class_names

try:
    model, CLASS_NAMES = load_model()
    print(f"✅  Model loaded — {len(CLASS_NAMES)} classes, device={device}")
except FileNotFoundError:
    print(f"❌  File tidak ditemukan: {MODEL_PATH}")
    model, CLASS_NAMES = None, DEFAULT_CLASS_NAMES
except RuntimeError as exc:
    print(f"❌  RuntimeError: {exc}")
    model, CLASS_NAMES = None, DEFAULT_CLASS_NAMES
except Exception as exc:
    print(f"❌  {type(exc).__name__}: {exc}")
    model, CLASS_NAMES = None, DEFAULT_CLASS_NAMES

# ── Preprocessing ──────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="WasteVision API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html tidak ditemukan.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/classify")
async def classify_image(files: List[UploadFile] = File(...)):
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maksimal 5 gambar yang bisa diunggah sekaligus.")

    if model is None:
        raise HTTPException(status_code=503, detail="Model belum berhasil dimuat. Cek terminal server.")

    results = []

    for file in files:
        if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} bukan format gambar yang didukung (Hanya JPG, PNG, WEBP).")

        raw = await file.read()
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File {file.filename} terlalu besar. Maksimum 5 MB.")

        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail=f"Gambar {file.filename} tidak bisa dibaca.")

        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        top3_values, top3_indices = torch.topk(probs, 3)
        top3 = [
            {"label": CLASS_NAMES[idx.item()], "confidence": round(val.item() * 100, 2)}
            for val, idx in zip(top3_values, top3_indices)
        ]

        results.append({
            "filename": file.filename,
            "predicted_label": top3[0]["label"],
            "confidence":      top3[0]["confidence"],
            "top3":            top3,
        })

    return JSONResponse({"results": results})
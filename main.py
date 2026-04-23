import io
from pathlib import Path
from typing import List

import torch
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── PATH ───────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_traced.pt"  # 🔥 pakai TorchScript

# ── CLASS NAMES ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes",
    "glass", "metal", "paper", "plastic", "shoes", "trash",
]

# ── CONFIG ─────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 Optional: stabilkan performa CPU kecil
torch.set_num_threads(1)

# ── LOAD MODEL (TorchScript) ───────────────────────────────────────────────────
try:
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    print(f"✅ TorchScript model loaded! device={device}")
except Exception as e:
    print(f"❌ Gagal load model: {e}")
    model = None

# ── PREPROCESS ─────────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── FASTAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="WasteVision API", version="1.0.0")

# Static files (CSS, JS, dll)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ── ROUTE: FRONTEND ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html tidak ditemukan.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

# ── ROUTE: CLASSIFY ────────────────────────────────────────────────────────────
@app.post("/classify")
async def classify_image(files: List[UploadFile] = File(...)):
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maksimal 5 gambar.")

    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")

    results = []

    for file in files:
        # Validasi format
        if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} bukan format valid (JPG/PNG/WEBP)."
            )

        # Validasi ukuran
        raw = await file.read()
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} terlalu besar (maks 5MB)."
            )

        # Load image
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Gambar {file.filename} tidak bisa dibaca."
            )

        # Preprocess
        tensor = preprocess(img).unsqueeze(0).to(device)

        # 🔥 INFERENCE CEPAT
        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # Top 3
        top3_values, top3_indices = torch.topk(probs, 3)

        top3 = [
            {
                "label": CLASS_NAMES[idx.item()],
                "confidence": round(val.item() * 100, 2)
            }
            for val, idx in zip(top3_values, top3_indices)
        ]

        results.append({
            "filename": file.filename,
            "predicted_label": top3[0]["label"],
            "confidence": top3[0]["confidence"],
            "top3": top3,
        })

    return JSONResponse({"results": results})
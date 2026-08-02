
from __future__ import annotations

import sys
import io

if sys.stdout.encoding.upper() != "UTF-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding.upper() != "UTF-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import base64
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model import UNetPlusPlusResNeXt50
from src.utils import IMAGENET_MEAN, IMAGENET_STD

NUM_CLASSES: int = 8
IGNORE_INDEX: int = 0
TILE_SIZE: int = 512
MAX_SELECTION_PX: int = 2560

CLASS_NAMES: dict[int, str] = {
    0: "ignore",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}

PALETTE: dict[int, tuple[int, int, int]] = {
    0: (27, 42, 74),
    1: (58, 70, 101),
    2: (255, 107, 26),
    3: (123, 132, 148),
    4: (95, 143, 199),
    5: (176, 137, 104),
    6: (79, 179, 169),
    7: (201, 180, 88),
}

class SegmentRequest(BaseModel):
    image_b64: str = Field(
        ...,
        description="Base64-encoded PNG of the satellite image (RGB, no labels)",
    )
    width_px: int = Field(..., ge=1, le=MAX_SELECTION_PX)
    height_px: int = Field(..., ge=1, le=MAX_SELECTION_PX)


class SegmentByCoordsRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    zoom: float = Field(..., ge=0, le=24)
    width_px: int = Field(..., ge=1, le=MAX_SELECTION_PX)
    height_px: int = Field(..., ge=1, le=MAX_SELECTION_PX)
    tile_size: int = Field(default=TILE_SIZE, ge=256, le=512)


class SegmentResponse(BaseModel):
    status: str
    mask_b64: Optional[str] = None
    overlay_b64: Optional[str] = None
    mask_url: Optional[str] = None
    original_b64: Optional[str] = None
    original_url: Optional[str] = None
    distribution: Optional[dict[str, float]] = None
    per_class_confidence: Optional[dict[str, float]] = None
    mean_confidence: Optional[float] = None
    duration_ms: float = 0
    note: Optional[str] = None


_model: Optional[torch.nn.Module] = None
_device: torch.device = torch.device("cpu")


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> torch.nn.Module:
    global _model, _device
    _device = torch.device(device)

    model = UNetPlusPlusResNeXt50(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)

    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=False)
    model.to(_device)
    model.eval()

    _model = model
    print(f"[infer] Model loaded from {checkpoint_path} on {_device}")
    return model


def get_model() -> torch.nn.Module:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


def _preprocess_tile(tile: Image.Image) -> torch.Tensor:
    arr = np.asarray(tile, dtype=np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0)


@torch.no_grad()
def segment_image(
    img: Image.Image,
    original_width: int,
    original_height: int,
    model: torch.nn.Module,
) -> dict:
    device = next(model.parameters()).device
    t0 = time.perf_counter()

    w, h = img.size
    pad_right = (TILE_SIZE - w % TILE_SIZE) % TILE_SIZE
    pad_bottom = (TILE_SIZE - h % TILE_SIZE) % TILE_SIZE
    padded_w, padded_h = w + pad_right, h + pad_bottom

    if pad_right > 0 or pad_bottom > 0:
        padded_img = Image.new("RGB", (padded_w, padded_h), (0, 0, 0))
        padded_img.paste(img, (0, 0))
    else:
        padded_img = img

    n_cols = padded_w // TILE_SIZE
    n_rows = padded_h // TILE_SIZE
    tiles_info: list[tuple[int, int, int, int, int, int]] = []
    all_tiles: list[torch.Tensor] = []

    for row in range(n_rows):
        for col in range(n_cols):
            x1 = col * TILE_SIZE
            y1 = row * TILE_SIZE
            x2 = x1 + TILE_SIZE
            y2 = y1 + TILE_SIZE

            tile = padded_img.crop((x1, y1, x2, y2))
            all_tiles.append(_preprocess_tile(tile).to(device))
            tiles_info.append((row, col, x1, y1, x2, y2))

    logit_accum = np.zeros((NUM_CLASSES, padded_h, padded_w), dtype=np.float32)

    for (row, col, x1, y1, x2, y2), tile_tensor in zip(tiles_info, all_tiles):
        logits = model(tile_tensor)
        probs = torch.softmax(logits, dim=1)
        logit_accum[:, y1:y2, x1:x2] = probs.squeeze(0).cpu().numpy()

    full_mask = np.argmax(logit_accum, axis=0).astype(np.uint8)

    if pad_right > 0:
        full_mask[:, w:] = IGNORE_INDEX
    if pad_bottom > 0:
        full_mask[h:, :] = IGNORE_INDEX

    mask = full_mask[:original_height, :original_width]

    conf_map = np.max(logit_accum, axis=0)[:original_height, :original_width]
    valid = mask != IGNORE_INDEX
    mean_conf = float(conf_map[valid].mean()) if valid.any() else 0.0

    mask_b64 = _mask_to_png_b64(mask)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    original_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    overlay_b64 = _build_overlay_b64(img, mask)

    total_valid = int(valid.sum())
    distribution: dict[str, float] = {}
    per_class_conf: dict[str, float] = {}
    for cls_idx in range(1, NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_idx]
        px_count = int((mask == cls_idx).sum())
        distribution[cls_name] = round(px_count / max(total_valid, 1), 4)
        cls_pixels = conf_map[mask == cls_idx]
        per_class_conf[cls_name] = round(float(cls_pixels.mean()), 4) if cls_pixels.size > 0 else 0.0

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    print(
        f"[infer] Segmented {original_width}x{original_height} px "
        f"> {n_rows}x{n_cols} = {len(all_tiles)} tiles in {elapsed_ms} ms"
    )

    return {
        "status": "ok",
        "mask_b64": mask_b64,
        "overlay_b64": overlay_b64,
        "distribution": distribution,
        "per_class_confidence": per_class_conf,
        "mean_confidence": round(mean_conf, 4),
        "duration_ms": elapsed_ms,
        "note": None,
    }


def _mask_to_png_b64(mask: np.ndarray) -> str:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in PALETTE.items():
        rgb[mask == cls_idx] = color
    img = Image.fromarray(rgb, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_overlay_b64(original: Image.Image, mask: np.ndarray) -> str:
    base = original.convert("RGBA")
    h, w = mask.shape

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    alpha = 140
    for cls_idx, color in PALETTE.items():
        if cls_idx == 0:
            continue
        overlay[mask == cls_idx] = (*color, alpha)

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    blended = Image.alpha_composite(base, overlay_img)

    buf = io.BytesIO()
    blended.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_pil(b64_str: str) -> Image.Image:
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


app = FastAPI(title="Urbaneuron Inference", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_MAPBOX_STATIC_MAX: int = 1280


def _fetch_satellite_image(
    lng: float, lat: float, zoom: float, width_px: int, height_px: int,
) -> Image.Image:
    token = os.environ.get("MAPBOX_TOKEN")
    if not token:
        raise HTTPException(
            status_code=500,
            detail="MAPBOX_TOKEN environment variable is not set",
        )

    cols = math.ceil(width_px / _MAPBOX_STATIC_MAX)
    rows = math.ceil(height_px / _MAPBOX_STATIC_MAX)
    tile_w = math.ceil(width_px / cols)
    tile_h = math.ceil(height_px / rows)

    stitched = Image.new("RGB", (tile_w * cols, tile_h * rows))

    metres_per_px = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

    for row in range(rows):
        for col in range(cols):
            cx_px = col * tile_w + tile_w / 2 - width_px / 2
            cy_px = row * tile_h + tile_h / 2 - height_px / 2

            offset_lng = cx_px * metres_per_px / (111_320 * math.cos(math.radians(lat)))
            offset_lat = -cy_px * metres_per_px / 111_320

            tile_lng = lng + offset_lng
            tile_lat = lat + offset_lat

            tw = min(tile_w, width_px - col * tile_w)
            th = min(tile_h, height_px - row * tile_h)

            req_w = min(tw, _MAPBOX_STATIC_MAX)
            req_h = min(th, _MAPBOX_STATIC_MAX)

            url = (
                "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
                f"{tile_lng},{tile_lat},{zoom},0,0/"
                f"{req_w}x{req_h}@2x"
                f"?access_token={token}"
            )

            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                detail = resp.text[:200] if resp.text else "(no body)"
                raise HTTPException(
                    status_code=502,
                    detail=f"Mapbox API returned {resp.status_code}: {detail}",
                )

            tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            actual_w, actual_h = tile_img.size
            if actual_w != tw or actual_h != th:
                tile_img = tile_img.resize((tw, th), Image.LANCZOS)

            stitched.paste(tile_img, (col * tile_w, row * tile_h))

    return stitched.crop((0, 0, width_px, height_px))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/infer", response_model=SegmentResponse)
def infer_by_coords(req: SegmentByCoordsRequest):
    t0 = time.perf_counter()

    if _model is None:
        return SegmentResponse(
            status="error",
            note="Model not loaded - ensure checkpoints/best.pt exists and restart",
            duration_ms=round((time.perf_counter() - t0) * 1000),
        )

    try:
        img = _fetch_satellite_image(
            req.lng, req.lat, req.zoom, req.width_px, req.height_px,
        )
    except HTTPException:
        raise
    except Exception as e:
        return SegmentResponse(
            status="error",
            note=f"Failed to fetch satellite image from Mapbox: {e}",
            duration_ms=round((time.perf_counter() - t0) * 1000),
        )

    try:
        result = segment_image(img, req.width_px, req.height_px, _model)
        if result.get("mask_b64"):
            result["mask_url"] = f"data:image/png;base64,{result['mask_b64']}"
        if result.get("original_b64"):
            result["original_url"] = f"data:image/png;base64,{result['original_b64']}"
        return SegmentResponse(**result)
    except Exception as e:
        msg = repr(str(e))[:250]
        return SegmentResponse(
            status="error",
            note=f"Inference failed | {type(e).__name__}: {msg}",
            duration_ms=round((time.perf_counter() - t0) * 1000),
        )


@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest):
    t0 = time.perf_counter()

    if _model is None:
        return SegmentResponse(
            status="error",
            note="Model not loaded - ensure checkpoints/best.pt exists and restart",
            duration_ms=round((time.perf_counter() - t0) * 1000),
        )

    try:
        img = _b64_to_pil(req.image_b64)
    except Exception as e:
        return SegmentResponse(
            status="error",
            note=f"Failed to decode image: {e}",
            duration_ms=round((time.perf_counter() - t0) * 1000),
        )

    if img.size != (req.width_px, req.height_px):
        return SegmentResponse(
            status="error",
            note=f"Image size {img.size} does not match declared {req.width_px}x{req.height_px}",
            duration_ms=round((time.perf_counter() - t0) * 1000),
        )

    try:
        result = segment_image(img, req.width_px, req.height_px, _model)
        return SegmentResponse(**result)
    except Exception as e:
        msg = repr(str(e))[:250]
        return SegmentResponse(
            status="error",
            note=f"Inference failed | {type(e).__name__}: {msg}",
            duration_ms=round((time.perf_counter() - t0) * 1000),
        )


if __name__ == "__main__":
    import uvicorn

    chkpt = os.environ.get("CHECKPOINT_PATH", "checkpoints/best.pt")
    device = os.environ.get(
        "INFER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )

    if Path(chkpt).exists():
        load_model(chkpt, device=device)
    else:
        print(
            f"[infer] WARNING: checkpoint not found at {chkpt} "
            f"- running without model"
        )

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
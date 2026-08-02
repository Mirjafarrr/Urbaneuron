# Urbaneuron

Urban satellite segmentation — U-Net++ with a ResNeXt-50 encoder, trained on LoveDA for pixel-level land-cover mapping of buildings, roads, water, and vegetation.

## Architecture

- **Model:** U-Net++ (nested dense skip-connection decoder) with a pretrained ResNeXt-50 encoder (~51M parameters).
- **Input:** 512×512 RGB satellite tiles at ~0.3 m/px (LoveDA, tiled from 1024×1024).
- **Output:** 8-class per-pixel segmentation (ignore, background, building, road, water, barren, forest, agriculture).
- **Loss:** CrossEntropy + Dice, inverse-frequency class weights, ignore index for padding/no-data.
- **Training:** 85% Urban / 15% Rural tile mix, 50/50 Urban/Rural validation split at source-image level.

## Project Structure

```
Urbaneuron/
├── main.py                  # Training entrypoint
├── inference_server.py      # FastAPI inference server (Docker / local)
├── Dockerfile               # Containerised inference
├── config/pipeline.yaml     # Hyperparameters
├── pyproject.toml           # Python dependencies (uv)
├── src/
│   ├── model.py             # U-Net++ (ResNeXt-50 encoder)
│   ├── engine.py            # Training loop (TF32, AMP)
│   ├── losses.py            # CE + Dice combo loss
│   ├── metrics.py           # mIoU, per-class IoU, pixel accuracy
│   ├── dataset.py           # LoveDA tile dataset + dataloaders
│   └── utils/               # Config loading, checkpointing, seed
├── scripts/                 # Data pipeline & utilities
│   ├── 01_crop_tiles.py
│   ├── 02_train_test_val_split.py
│   ├── 03_data_quality.py
│   └── check_env.py
├── frontend/                # React + Mapbox GL JS (Vercel)
├── checkpoints/             # Saved model weights (gitignored)
├── data/                    # LoveDA tiles (gitignored)
├── tests/                   # Unit tests
└── notebooks/               # Exploration notebooks
```

## Data

LoveDA — 5,987 satellite images (1024×1024, 0.3 m/px) covering Nanjing, Changzhou, and Wuhan. Each image is split into four non-overlapping 512×512 tiles. The training pool uses an 85/15 Urban/Rural mix; validation uses a fixed 50/50 split at the source-image level to avoid tile leakage.

## Setup

```bash
uv sync
```

## Training

**Local (consumer GPU):**

```bash
python main.py --batch-size 4 --epochs 30 --num-workers 6
```

**Cloud (A100 / 4090):**

```bash
python main.py
```

All hyperparameters live in `config/pipeline.yaml`. The encoder learning rate is kept low (5e-5) to preserve pretrained ImageNet features; the decoder trains from scratch at 10× the encoder rate (5e-4).

## Inference Server

Start the FastAPI inference server locally:

```bash
python inference_server.py
```

Or via Docker:

```bash
docker build -t urbaneuron-infer . && docker run --gpus all -p 8000:8000 urbaneuron-infer
```

The server exposes three endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check + model load status |
| POST | `/infer` | Coordinate-based inference (lat, lng, zoom, width, height) |
| POST | `/segment` | Direct image inference (base64 PNG) |

Coordinate-based requests fetch satellite imagery from Mapbox Static Images API, pad to 512 multiples, tile, run segmentation, stitch, and return the mask, overlay, and per-class statistics.

## Frontend

React + Mapbox GL JS site deployed on Vercel. Users draw a bounding box (up to 2560×2560 px), the selection is sent to the inference server, and results are displayed with a before/after slider, class-colour overlay, and download options.

## Config Reference

| Setting | Value |
|---|---|
| Architecture | U-Net++ / ResNeXt-50 |
| Input size | 512×512 RGB |
| Classes | 8 (ignore, background, building, road, water, barren, forest, agriculture) |
| Train mix | 85% Urban / 15% Rural |
| Val split | 50/50 Urban/Rural |
| Loss | CrossEntropy (0.7) + Dice (0.3) |
| Epochs | 50 (cloud) / 30 (local) |
| Batch size | 32 (A100) / 4 (local) |
| Encoder LR | 5e-5 |
| Decoder LR | 5e-4 |
| Mixed precision | torch.cuda.amp + TF32 |
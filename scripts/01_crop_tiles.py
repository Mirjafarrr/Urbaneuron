
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import rasterio
from tqdm import tqdm


SRC_SIZE: int = 1024
TILE_SIZE: int = 512
GRID_ROWS: int = 2
GRID_COLS: int = 2

IMG_EXTS: Set[str] = {".png", ".tif", ".tiff"}
OUTPUT_BASE: Path = Path("data/processed")

PROCESSING_ORDER: List[Tuple[str, str]] = [
    ("Rural", "Train"),
    ("Urban", "Train"),
    ("Rural", "Val"),
    ("Urban", "Val"),
    ("Rural", "Test"),
    ("Urban", "Test"),
]

TILE_MANIFEST_COLS = [
    "global_id",
    "region",
    "orig_split",
    "source_stem",
    "source_path",
    "tile_path",
    "mask_path",
    "tile_row",
    "tile_col",
    "status",
]




def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def _validate_1024x1024(path: Path, arr: np.ndarray) -> bool:
    h, w = arr.shape[:2]
    if h != SRC_SIZE or w != SRC_SIZE:
        print(
            f"[WARN]  {path}  shape=({h},{w})  expected=({SRC_SIZE},{SRC_SIZE})  — skipping",
            file=sys.stderr,
        )
        return False
    return True


def _read_as_uint8(path: Path) -> np.ndarray:
    with rasterio.open(str(path)) as src:
        data: np.ndarray = src.read()
    if data.shape[0] == 1:
        return np.ascontiguousarray(data[0])
    rgb = data[:3].transpose(1, 2, 0)
    return np.ascontiguousarray(rgb)


def _crop_tile(img: np.ndarray, row: int, col: int) -> np.ndarray:
    y0, y1 = row * TILE_SIZE, (row + 1) * TILE_SIZE
    x0, x1 = col * TILE_SIZE, (col + 1) * TILE_SIZE
    return img[y0:y1, x0:x1]


def _write_image(path: Path, arr: np.ndarray) -> None:
    with rasterio.open(
        str(path), "w", driver="PNG",
        height=arr.shape[0], width=arr.shape[1],
        count=3, dtype="uint8",
    ) as dst:
        dst.write(arr.transpose(2, 0, 1))


def _write_mask(path: Path, arr: np.ndarray) -> None:
    with rasterio.open(
        str(path), "w", driver="PNG",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="uint8",
    ) as dst:
        dst.write(arr, 1)




def _find_region_root(raw_dir: Path, region: str) -> Path:
    matches: List[Path] = []
    for entry in raw_dir.rglob("*"):
        if entry.is_dir() and entry.name.lower() == region.lower():
            matches.append(entry)
    if not matches:
        print(
            f"[FATAL]  No directory matching '{region}' found under {raw_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    return sorted(matches, key=lambda p: len(p.parts))[0]


def _find_split_dir(region_root: Path, split_name: str) -> Path:
    for entry in sorted(region_root.iterdir()):
        if entry.is_dir() and entry.name.lower() == split_name.lower():
            return entry
    print(
        f"[FATAL]  No '{split_name}' directory found under {region_root}",
        file=sys.stderr,
    )
    sys.exit(1)


def _collect_source_pairs(
    img_dir: Path, mask_dir: Optional[Path],
) -> List[Tuple[Path, Optional[Path]]]:
    if mask_dir is None:
        return [
            (p, None)
            for p in sorted(img_dir.iterdir())
            if p.is_file() and _is_image_file(p)
        ]
    mask_stems: Set[str] = {
        p.stem for p in mask_dir.iterdir() if p.is_file() and _is_image_file(p)
    }
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or not _is_image_file(img_path):
            continue
        stem = img_path.stem
        if stem not in mask_stems:
            print(f"[WARN]  No mask for: {img_path}", file=sys.stderr)
            continue
        mask_path = mask_dir / f"{stem}{img_path.suffix}"
        if not mask_path.is_file():
            print(f"[WARN]  Missing: {mask_path}", file=sys.stderr)
            continue
        pairs.append((img_path, mask_path))
    return pairs




def _tile_one_source(
    img_path: Path,
    mask_path: Optional[Path],
    region: str,
    orig_split: str,
    out_base: Path,
    start_id: int,
) -> Tuple[List[dict], int]:
    stem = img_path.stem
    tile_rows: List[dict] = []
    next_id = start_id

    try:
        image = _read_as_uint8(img_path)
        if not _validate_1024x1024(img_path, image):
            return tile_rows, next_id

        mask: Optional[np.ndarray] = None
        if mask_path is not None:
            mask = _read_as_uint8(mask_path)
            if not _validate_1024x1024(mask_path, mask):
                return tile_rows, next_id

        out_img_dir = out_base / region / orig_split / "images_png"
        out_mask_dir = out_base / region / orig_split / "masks_png" if mask_path else None
        out_img_dir.mkdir(parents=True, exist_ok=True)
        if out_mask_dir:
            out_mask_dir.mkdir(parents=True, exist_ok=True)

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                tile_id = next_id
                next_id += 1

                img_tile = _crop_tile(image, r, c)
                img_out = out_img_dir / f"{tile_id}.png"
                _write_image(img_out, img_tile)

                mask_out_str = ""
                if mask is not None and out_mask_dir is not None:
                    mask_tile = _crop_tile(mask, r, c)
                    mask_out = out_mask_dir / f"{tile_id}.png"
                    _write_mask(mask_out, mask_tile)
                    mask_out_str = str(mask_out)

                tile_rows.append({
                    "global_id": tile_id,
                    "region": region,
                    "orig_split": orig_split,
                    "source_stem": stem,
                    "source_path": str(img_path),
                    "tile_path": str(img_out),
                    "mask_path": mask_out_str,
                    "tile_row": r,
                    "tile_col": c,
                    "status": "ok",
                })

    except Exception as exc:
        print(f"[WARN]  {img_path}: {exc}", file=sys.stderr)
        tile_rows = [{
            "global_id": start_id,
            "region": region,
            "orig_split": orig_split,
            "source_stem": stem,
            "source_path": str(img_path),
            "tile_path": "",
            "mask_path": "",
            "tile_row": -1,
            "tile_col": -1,
            "status": "skipped_ioerror",
        }]
        next_id = start_id + 1

    return tile_rows, next_id




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tile LoveDA 1024×1024 → 512×512 (no split — split is in script 02)."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                        help="Root directory containing LoveDA data (default: data/raw)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_BASE,
                        help="Output directory (default: data/processed)")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    output_base: Path = args.output_dir

    if not raw_dir.is_dir():
        print(f"[FATAL]  Raw directory not found: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  URBANEURON — 01_crop_tiles.py  (TILE ONLY)")
    print("=" * 60)
    print(f"  Raw directory    : {raw_dir}")
    print(f"  Output directory : {output_base}")
    print(f"  Tile size        : {TILE_SIZE}×{TILE_SIZE}  (grid: {GRID_ROWS}×{GRID_COLS})")
    print("=" * 60)

    output_base.mkdir(parents=True, exist_ok=True)
    all_tile_rows: List[dict] = []
    next_global_id: int = 0

    for region, orig_split in PROCESSING_ORDER:
        region_root = _find_region_root(raw_dir, region)
        split_dir = _find_split_dir(region_root, orig_split)

        img_dir = split_dir / "images_png"
        if not img_dir.is_dir():
            print(f"[WARN]  No images_png/ in {split_dir} — skipping", file=sys.stderr)
            continue
        mask_dir = split_dir / "masks_png"
        mask_dir = mask_dir if mask_dir.is_dir() else None

        sources = _collect_source_pairs(img_dir, mask_dir)
        has_mask = "with" if mask_dir else "NO"
        print(f"  Found  {region}/{orig_split:<5}  {len(sources):>5d} images  ({has_mask} masks)")

        desc = f"Tiling  {region}/{orig_split}"
        for img_path, mask_path in tqdm(sources, desc=desc, unit="img"):
            rows, next_global_id = _tile_one_source(
                img_path, mask_path, region, orig_split, output_base, next_global_id
            )
            all_tile_rows.extend(rows)

    manifest_path = output_base / "tile_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TILE_MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(all_tile_rows)

    ok_count = sum(1 for r in all_tile_rows if r["status"] == "ok")
    skip_count = sum(1 for r in all_tile_rows if r["status"] != "ok")

    print()
    print("=" * 60)
    print("  TILING COMPLETE")
    print("=" * 60)
    print(f"  Total tiles written  : {len(all_tile_rows)}")
    print(f"    OK                  : {ok_count}")
    print(f"    Skipped             : {skip_count}")
    print(f"  Final global_id      : {next_global_id}")
    print(f"  Manifest             : {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
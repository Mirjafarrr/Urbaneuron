
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm


DEFAULT_VAL_FRAC: float = 0.10
DEFAULT_TRAIN_URBAN_FRAC: float = 0.85
DEFAULT_SEED: int = 42
TILES_PER_SOURCE: int = 4

MANIFEST_HEADER = [
    "source_stem",
    "global_id",
    "region",
    "orig_split",
    "assigned_split",
    "tile_path",
    "mask_path",
]



def _load_tile_manifest(
    manifest_path: Path,
) -> Tuple[
    Dict[str, List[dict]],
    Dict[str, List[dict]],
    Dict[str, List[dict]],
    Dict[str, List[dict]],
]:
    urban_trainval: Dict[str, List[dict]] = defaultdict(list)
    rural_trainval: Dict[str, List[dict]] = defaultdict(list)
    urban_test: Dict[str, List[dict]] = defaultdict(list)
    rural_test: Dict[str, List[dict]] = defaultdict(list)

    with open(manifest_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("status", "") != "ok":
                continue
            region = row.get("region", "")
            orig_split = row.get("orig_split", "")
            stem = row.get("source_stem", "")
            if not stem:
                continue

            if region == "Urban":
                if orig_split in ("Train", "Val"):
                    urban_trainval[stem].append(row)
                elif orig_split == "Test":
                    urban_test[stem].append(row)
            elif region == "Rural":
                if orig_split in ("Train", "Val"):
                    rural_trainval[stem].append(row)
                elif orig_split == "Test":
                    rural_test[stem].append(row)

    return urban_trainval, rural_trainval, urban_test, rural_test


def _carve_validation(
    urban_stems: List[str],
    rural_stems: List[str],
    val_frac: float,
    rng: random.Random,
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    n_val_per_class = max(1, round(min(len(urban_stems), len(rural_stems)) * val_frac))

    urban_shuffled = sorted(urban_stems)
    rng.shuffle(urban_shuffled)
    val_urban = set(urban_shuffled[:n_val_per_class])
    train_pool_urban = set(urban_shuffled[n_val_per_class:])

    rural_shuffled = sorted(rural_stems)
    rng.shuffle(rural_shuffled)
    val_rural = set(rural_shuffled[:n_val_per_class])
    train_pool_rural = set(rural_shuffled[n_val_per_class:])

    return val_urban, val_rural, train_pool_urban, train_pool_rural



_TILE_MISSING_WARNED: Set[str] = set()


def _copy_tiles(
    stems: Set[str],
    tile_groups: Dict[str, List[dict]],
    dest_img_dir: Path,
    dest_mask_dir: Optional[Path],
    assigned_split: str,
    start_local_id: int,
    no_copy: bool,
) -> Tuple[List[dict], int]:
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    if dest_mask_dir:
        dest_mask_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    local_id: int = start_local_id
    for stem in tqdm(sorted(stems), desc=f"  → {assigned_split}", unit="src"):
        tiles = tile_groups.get(stem, [])
        for tile in tiles:
            src_img = Path(tile["tile_path"])
            ext = src_img.suffix.lower()
            local_name = f"{local_id}{ext}"
            dst_img = dest_img_dir / local_name

            if not no_copy:
                if src_img.is_file():
                    shutil.copy2(src_img, dst_img)
                else:
                    if src_img.as_posix() not in _TILE_MISSING_WARNED:
                        print(f"[WARN]  Missing tile: {src_img}", file=sys.stderr)
                        _TILE_MISSING_WARNED.add(src_img.as_posix())
                    local_id += 1
                    continue

            mask_path_str = ""
            src_mask = Path(tile["mask_path"]) if tile.get("mask_path") else None
            if src_mask and dest_mask_dir:
                mask_ext = src_mask.suffix.lower()
                dst_mask = dest_mask_dir / f"{local_id}{mask_ext}"
                if not no_copy:
                    if src_mask.is_file():
                        shutil.copy2(src_mask, dst_mask)
                        mask_path_str = str(dst_mask)
                    else:
                        if src_mask.as_posix() not in _TILE_MISSING_WARNED:
                            print(f"[WARN]  Missing mask: {src_mask}", file=sys.stderr)
                            _TILE_MISSING_WARNED.add(src_mask.as_posix())
                        mask_path_str = str(src_mask)
                else:
                    mask_path_str = str(src_mask)

            rows.append({
                "source_stem": stem,
                "global_id": tile.get("global_id", ""),
                "region": tile.get("region", ""),
                "orig_split": tile.get("orig_split", ""),
                "assigned_split": assigned_split,
                "tile_path": str(dst_img) if not no_copy else str(src_img),
                "mask_path": mask_path_str,
            })

            local_id += 1

    return rows, local_id


def _copy_test_tiles(
    stems: Set[str],
    tile_groups: Dict[str, List[dict]],
    output_base: Path,
    no_copy: bool,
) -> List[dict]:
    img_dir = output_base / "test" / "images_png"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    local_id: int = 0
    for stem in tqdm(sorted(stems), desc="  → test", unit="src"):
        tiles = tile_groups.get(stem, [])
        for tile in tiles:
            src_img = Path(tile["tile_path"])
            ext = src_img.suffix.lower()
            dst_img = img_dir / f"{local_id}{ext}"

            if not no_copy:
                if src_img.is_file():
                    shutil.copy2(src_img, dst_img)
                else:
                    if src_img.as_posix() not in _TILE_MISSING_WARNED:
                        print(f"[WARN]  Missing test tile: {src_img}", file=sys.stderr)
                        _TILE_MISSING_WARNED.add(src_img.as_posix())
                    local_id += 1
                    continue

            rows.append({
                "source_stem": stem,
                "global_id": tile.get("global_id", ""),
                "region": tile.get("region", ""),
                "orig_split": tile.get("orig_split", ""),
                "assigned_split": "test",
                "tile_path": str(dst_img) if not no_copy else str(src_img),
                "mask_path": "",
            })

            local_id += 1

    return rows


def _write_manifest(path: Path, rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)




def main() -> None:
    parser = argparse.ArgumentParser(
        description="85/15 train + 50/50 val split → data/final/"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/tile_manifest.csv"),
        help="Path to tile_manifest.csv from 01_crop_tiles.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/final"),
        help="Output directory (default: data/final)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help=(
            f"Fraction of the SMALLER class reserved for validation — "
            f"both classes contribute this many sources (default: {DEFAULT_VAL_FRAC})"
        ),
    )
    parser.add_argument(
        "--train-urban-frac",
        type=float,
        default=DEFAULT_TRAIN_URBAN_FRAC,
        help=f"Urban proportion in the training pool (default: {DEFAULT_TRAIN_URBAN_FRAC})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Write manifests only; skip physical tile copies",
    )
    args = parser.parse_args()

    manifest_path: Path = args.manifest
    output_dir: Path = args.output_dir
    val_frac: float = args.val_frac
    train_urban_frac: float = args.train_urban_frac
    seed: int = args.seed
    no_copy: bool = args.no_copy

    if not manifest_path.is_file():
        print(f"[FATAL]  Manifest not found: {manifest_path}", file=sys.stderr)
        print("         Run 01_crop_tiles.py first.", file=sys.stderr)
        sys.exit(1)

    if not 0.0 < val_frac < 1.0:
        print(f"[FATAL]  --val-frac must be in (0.0, 1.0), got {val_frac}", file=sys.stderr)
        sys.exit(1)

    if not 0.0 < train_urban_frac < 1.0:
        print(f"[FATAL]  --train-urban-frac must be in (0.0, 1.0), got {train_urban_frac}", file=sys.stderr)
        sys.exit(1)

    train_rural_frac = 1.0 - train_urban_frac
    rng = random.Random(seed)

    print()
    print("=" * 70)
    print("  URBANEURON — 02_train_test_val_split.py")
    print("=" * 70)
    print(f"  Manifest           : {manifest_path}")
    print(f"  Output directory   : {output_dir}")
    print(f"  Val fraction       : {val_frac:.0%} of each class")
    print(f"  Train Urban/Rural  : {train_urban_frac:.0%} / {train_rural_frac:.0%}")
    print(f"  Seed               : {seed}")
    if no_copy:
        print("  Mode               : MANIFEST-ONLY (no physical copies)")
    print("=" * 70)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print("\n[Phase 1]  Loading tile manifest ...")
    urban_trainval, rural_trainval, urban_test, rural_test = _load_tile_manifest(manifest_path)

    n_urban_src = len(urban_trainval)
    n_rural_src = len(rural_trainval)
    print(f"  Urban  Train+Val sources : {n_urban_src}  ({n_urban_src * TILES_PER_SOURCE} tiles)")
    print(f"  Rural  Train+Val sources : {n_rural_src}  ({n_rural_src * TILES_PER_SOURCE} tiles)")
    print(f"  Urban  Test       sources : {len(urban_test)}  ({len(urban_test) * TILES_PER_SOURCE} tiles)")
    print(f"  Rural  Test       sources : {len(rural_test)}  ({len(rural_test) * TILES_PER_SOURCE} tiles)")

    if not urban_trainval:
        print("[FATAL]  No Urban Train+Val tiles found in manifest.", file=sys.stderr)
        sys.exit(1)
    if not rural_trainval:
        print("[FATAL]  No Rural Train+Val tiles found in manifest.", file=sys.stderr)
        sys.exit(1)

    n_val_per = max(1, round(min(n_urban_src, n_rural_src) * val_frac))
    print(f"\n[Phase 2]  Carving validation — {n_val_per} sources per class (50/50 by source count)")

    urban_stems = sorted(urban_trainval.keys())
    rural_stems = sorted(rural_trainval.keys())

    val_urban, val_rural, train_pool_urban, train_pool_rural = _carve_validation(
        urban_stems, rural_stems, val_frac, rng
    )

    print(f"  Val    Urban: {len(val_urban)} sources  ({len(val_urban) * TILES_PER_SOURCE} tiles)")
    print(f"  Val    Rural: {len(val_rural)} sources  ({len(val_rural) * TILES_PER_SOURCE} tiles)")
    print(f"  Val    Total: {len(val_urban) + len(val_rural)} sources  "
          f"({(len(val_urban) + len(val_rural)) * TILES_PER_SOURCE} tiles)")

    print(f"\n[Phase 3]  Building training pool ({train_urban_frac:.0%} Urban / {train_rural_frac:.0%} Rural) ...")

    n_train_urban = len(train_pool_urban)
    n_train_rural_needed = max(0, round(n_train_urban * train_rural_frac / train_urban_frac))

    rural_train_stems = sorted(train_pool_rural)
    rng.shuffle(rural_train_stems)
    train_rural = set(rural_train_stems[:n_train_rural_needed])
    unused_rural = set(rural_train_stems[n_train_rural_needed:])

    print(f"  Train  Urban       : {n_train_urban} sources  ({n_train_urban * TILES_PER_SOURCE} tiles)")
    print(f"  Train  Rural       : {len(train_rural)} sources  ({len(train_rural) * TILES_PER_SOURCE} tiles)")
    print(f"  Train  Total       : {n_train_urban + len(train_rural)} sources  "
          f"({(n_train_urban + len(train_rural)) * TILES_PER_SOURCE} tiles)")
    if unused_rural:
        print(f"  UNUSED Rural       : {len(unused_rural)} sources  ({len(unused_rural) * TILES_PER_SOURCE} tiles)")

    print("\n[Phase 4]  Copying validation tiles (50/50 Urban/Rural) ...")
    all_trainval = {**urban_trainval, **rural_trainval}

    val_rows, _ = _copy_tiles(
        val_urban | val_rural,
        all_trainval,
        output_dir / "val" / "images_png",
        output_dir / "val" / "masks_png",
        "val",
        start_local_id=0,
        no_copy=no_copy,
    )
    _write_manifest(output_dir / "val" / "manifest.csv", val_rows)
    print(f"    Wrote {len(val_rows)} tile rows → {output_dir / 'val' / 'manifest.csv'}")

    print("\n[Phase 5]  Copying training tiles (85/15 Urban/Rural) ...")

    train_rows, _ = _copy_tiles(
        train_pool_urban | train_rural,
        all_trainval,
        output_dir / "train" / "images_png",
        output_dir / "train" / "masks_png",
        "train",
        start_local_id=0,
        no_copy=no_copy,
    )
    _write_manifest(output_dir / "train" / "manifest.csv", train_rows)
    print(f"    Wrote {len(train_rows)} tile rows → {output_dir / 'train' / 'manifest.csv'}")

    all_test_urban = set(urban_test.keys())
    all_test_rural = set(rural_test.keys())
    all_test = all_test_urban | all_test_rural

    print(f"\n[Phase 6]  Copying test tiles ({len(all_test)} sources, no masks) ...")
    test_rows = _copy_test_tiles(
        all_test,
        {**urban_test, **rural_test},
        output_dir,
        no_copy=no_copy,
    )
    _write_manifest(output_dir / "test" / "manifest.csv", test_rows)
    print(f"    Wrote {len(test_rows)} tile rows → {output_dir / 'test' / 'manifest.csv'}")

    actual_train_urban = len(train_pool_urban)
    actual_train_rural = len(train_rural)
    train_size = actual_train_urban + actual_train_rural
    actual_train_pct = actual_train_urban / train_size * 100 if train_size > 0 else 0

    print()
    print("=" * 70)
    print("  SPLIT COMPLETE")
    print("=" * 70)
    print(f"  Output directory      : {output_dir}")
    print(f"  Train tiles           : {len(train_rows)}")
    print(f"    Actual Urban/Rural  : {actual_train_urban} / {actual_train_rural} sources")
    print(f"    Actual Urban %      : {actual_train_pct:.1f}%")
    print(f"  Val   tiles           : {len(val_rows)}")
    print(f"    Urban/Rural         : {len(val_urban)} / {len(val_rural)} sources")
    print(f"  Test  tiles           : {len(test_rows)}")
    if unused_rural:
        print(f"  UNUSED Rural          : {len(unused_rural)} sources "
              f"({len(unused_rural) * TILES_PER_SOURCE} tiles)")
    print(f"  Total written         : {len(train_rows) + len(val_rows) + len(test_rows)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
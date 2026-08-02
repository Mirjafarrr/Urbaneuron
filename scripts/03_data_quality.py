
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image


EXPECTED_TILE_SIZE: Tuple[int, int] = (512, 512)
EXPECTED_TILES_PER_SOURCE: int = 4
LOVEDA_CLASSES: Set[int] = {0, 1, 2, 3, 4, 5, 6, 7}
LOVEDA_CLASS_NAMES: Dict[int, str] = {
    0: "Ignore / No-data",
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren",
    6: "Forest",
    7: "Agriculture",
}
NON_IGNORE_CLASSES: Set[int] = {1, 2, 3, 4, 5, 6, 7}

EXPECTED_SPLITS: List[str] = ["train", "val", "test"]

REQUIRED_MANIFEST_COLS: Set[str] = {
    "source_stem",
    "global_id",
    "region",
    "orig_split",
    "assigned_split",
    "tile_path",
    "mask_path",
}

EXPECTED_TRAIN_URBAN_FRAC: float = 0.85


_SEVERITY_ORDER: Dict[str, int] = {"FATAL": 0, "ERROR": 1, "WARN": 2, "INFO": 3}


class CheckResult:

    def __init__(
        self,
        check_id: str,
        category: str,
        name: str,
        severity: str,
        passed: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.check_id = check_id
        self.category = category
        self.name = name
        self.severity = severity
        self.passed = passed
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "category": self.category,
            "name": self.name,
            "severity": self.severity,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


class QualityReport:

    def __init__(self) -> None:
        self.checks: List[CheckResult] = []
        self.per_split_stats: Dict[str, Dict[str, Any]] = {}
        self.class_distribution: Dict[str, Dict[int, int]] = {}
        self.start_time = time.time()

    def add(self, cr: CheckResult) -> None:
        self.checks.append(cr)

    def all_passed(self) -> bool:
        for c in self.checks:
            if c.severity in ("FATAL", "ERROR") and not c.passed:
                return False
        return True

    def text_summary(self) -> str:
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("  URBANEURON — Data Quality Report  (v13)")
        lines.append("=" * 72)
        elapsed = time.time() - self.start_time
        lines.append(f"  Ran in {elapsed:.1f}s  |  {len(self.checks)} checks executed")
        lines.append("")

        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"PASS": 0, "FAIL": 0})
        for c in self.checks:
            key = "PASS" if c.passed else "FAIL"
            counts[c.severity][key] += 1

        lines.append("  ┌──────────┬──────┬──────┐")
        lines.append("  │ Severity │ PASS │ FAIL │")
        lines.append("  ├──────────┼──────┼──────┤")
        for sev in ("FATAL", "ERROR", "WARN", "INFO"):
            p = counts[sev]["PASS"]
            f = counts[sev]["FAIL"]
            lines.append(f"  │ {sev:<8} │ {p:>4} │ {f:>4} │")
        lines.append("  └──────────┴──────┴──────┘")
        lines.append("")

        if self.all_passed():
            lines.append("  >>> OVERALL: PASS — all critical checks succeeded. <<<")
        else:
            lines.append("  >>> OVERALL: FAIL — one or more FATAL/ERROR checks failed. <<<")
        lines.append("")

        current_cat = ""
        for c in self.checks:
            if c.category != current_cat:
                current_cat = c.category
                lines.append(f"\n{'─' * 72}")
                lines.append(f"  {current_cat}")
                lines.append(f"{'─' * 72}")

            status = "PASS" if c.passed else "FAIL"
            marker = f"[{c.severity}]"
            lines.append(f"  {marker:<9} {status:<6} {c.check_id}  {c.name}")
            if c.message:
                lines.append(f"            {c.message}")

        if self.per_split_stats:
            lines.append(f"\n{'─' * 72}")
            lines.append("  Per-Split Statistics")
            lines.append(f"{'─' * 72}")
            for split_name, stats in self.per_split_stats.items():
                lines.append(f"\n  [{split_name}]")
                for k, v in stats.items():
                    lines.append(f"    {k}: {v}")

        if self.class_distribution:
            lines.append(f"\n{'─' * 72}")
            lines.append("  Class Distribution (non-ignore pixels)")
            lines.append(f"{'─' * 72}")
            for split_name, class_counts in self.class_distribution.items():
                total = sum(class_counts.values())
                lines.append(f"\n  [{split_name}]")
                lines.append(f"  {'Class':<20} {'Pixels':>12} {'Pct':>8}")
                lines.append(f"  {'-'*40}")
                for cls_id in sorted(class_counts.keys()):
                    name = LOVEDA_CLASS_NAMES.get(cls_id, f"Unknown({cls_id})")
                    cnt = class_counts[cls_id]
                    pct = (cnt / total * 100) if total > 0 else 0.0
                    lines.append(f"  {cls_id} {name:<17} {cnt:>12,} {pct:>7.2f}%")
                lines.append(f"  {'TOTAL':<20} {total:>12,}")

        lines.append(f"\n{'=' * 72}")
        return "\n".join(lines)

    def json_summary(self) -> Dict[str, Any]:
        return {
            "status": "PASS" if self.all_passed() else "FAIL",
            "elapsed_seconds": round(time.time() - self.start_time, 1),
            "total_checks": len(self.checks),
            "checks": [c.to_dict() for c in self.checks],
            "per_split_stats": self.per_split_stats,
            "class_distribution": {
                split: {str(k): v for k, v in cls.items()}
                for split, cls in self.class_distribution.items()
            },
        }




def _parse_manifest(
    csv_path: Path, report: QualityReport, check_id: str
) -> List[Dict[str, str]]:
    if not csv_path.is_file():
        report.add(
            CheckResult(
                check_id=check_id,
                category="1. Structural Integrity",
                name=f"Manifest file exists: {csv_path.name}",
                severity="FATAL",
                passed=False,
                message=f"File not found: {csv_path}",
            )
        )
        return []

    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                report.add(
                    CheckResult(
                        check_id=check_id,
                        category="1. Structural Integrity",
                        name=f"Manifest has header: {csv_path.name}",
                        severity="FATAL",
                        passed=False,
                        message="No header row found.",
                    )
                )
                return []
            rows = list(reader)
    except UnicodeDecodeError as e:
        report.add(
            CheckResult(
                check_id=check_id,
                category="1. Structural Integrity",
                name=f"Manifest UTF-8 encoding: {csv_path.name}",
                severity="ERROR",
                passed=False,
                message=f"Encoding error: {e}",
            )
        )
        return []
    except Exception as e:
        report.add(
            CheckResult(
                check_id=check_id,
                category="1. Structural Integrity",
                name=f"Manifest is valid CSV: {csv_path.name}",
                severity="FATAL",
                passed=False,
                message=f"Parse error: {e}",
            )
        )
        return []

    if reader.fieldnames:
        missing = REQUIRED_MANIFEST_COLS - set(reader.fieldnames)
        if missing:
            report.add(
                CheckResult(
                    check_id=check_id,
                    category="1. Structural Integrity",
                    name=f"Manifest required columns: {csv_path.name}",
                    severity="FATAL",
                    passed=False,
                    message=f"Missing columns: {missing}",
                )
            )
            return []

    report.add(
        CheckResult(
            check_id=f"{check_id}_parsed",
            category="1. Structural Integrity",
            name=f"Manifest parsed: {csv_path.name}",
            severity="INFO",
            passed=True,
            message=f"{len(rows)} rows",
        )
    )
    return rows


def _get_split_stems(rows: List[Dict[str, str]]) -> Set[str]:
    return {r.get("source_stem", "") for r in rows if r.get("source_stem", "").strip()}


def _get_tile_paths(rows: List[Dict[str, str]]) -> Set[str]:
    return {r.get("tile_path", "") for r in rows if r.get("tile_path", "").strip()}




def _validate_image_file(
    filepath: Path,
    report: QualityReport,
    check_prefix: str,
    category: str,
) -> Optional[np.ndarray]:
    if not filepath.is_file():
        report.add(
            CheckResult(
                check_id=f"{check_prefix}_missing",
                category=category,
                name=f"File exists: {filepath.name}",
                severity="FATAL",
                passed=False,
                message=f"Not found: {filepath}",
            )
        )
        return None

    if filepath.stat().st_size == 0:
        report.add(
            CheckResult(
                check_id=f"{check_prefix}_zero_byte",
                category=category,
                name=f"Non-zero file: {filepath.name}",
                severity="FATAL",
                passed=False,
                message=f"Zero-byte file: {filepath}",
            )
        )
        return None

    try:
        img = Image.open(filepath)
    except Exception as e:
        report.add(
            CheckResult(
                check_id=f"{check_prefix}_open",
                category=category,
                name=f"Image can be opened: {filepath.name}",
                severity="FATAL",
                passed=False,
                message=f"PIL open error: {e}",
            )
        )
        return None

    try:
        arr = np.array(img)
    except Exception as e:
        report.add(
            CheckResult(
                check_id=f"{check_prefix}_load",
                category=category,
                name=f"Pixel data loadable: {filepath.name}",
                severity="FATAL",
                passed=False,
                message=f"numpy conversion error: {e}",
            )
        )
        img.close()
        return None

    img.close()
    return arr


def _validate_mask_values(
    mask_arr: np.ndarray,
    filepath: Path,
    report: QualityReport,
    check_prefix: str,
) -> bool:
    all_ok = True

    if not np.issubdtype(mask_arr.dtype, np.integer):
        report.add(
            CheckResult(
                check_id=f"{check_prefix}_int_dtype",
                category="4. Mask Validation",
                name=f"Mask dtype is integer: {filepath.name}",
                severity="FATAL",
                passed=False,
                message=f"Non-integer dtype={mask_arr.dtype}",
            )
        )
        all_ok = False
        return all_ok

    unique_vals = set(np.unique(mask_arr).tolist())
    unexpected = unique_vals - LOVEDA_CLASSES
    if unexpected:
        report.add(
            CheckResult(
                check_id=f"{check_prefix}_invalid_values",
                category="4. Mask Validation",
                name=f"Only valid LoveDA values: {filepath.name}",
                severity="FATAL",
                passed=False,
                message=f"Unexpected values: {sorted(unexpected)}",
                details={"unexpected_values": sorted(unexpected)},
            )
        )
        all_ok = False

    return all_ok




def run_validation(
    data_dir: Path,
    full_scan: bool,
    target_split: Optional[str],
    batch_size: int,
    sample_frac: float,
    output_dir: Path,
) -> QualityReport:
    report = QualityReport()

    if not data_dir.is_dir():
        report.add(
            CheckResult(
                check_id="1.0",
                category="1. Structural Integrity",
                name="Data directory exists",
                severity="FATAL",
                passed=False,
                message=f"Not found: {data_dir}",
            )
        )
        return report
    report.add(
        CheckResult(
            check_id="1.0",
            category="1. Structural Integrity",
            name="Data directory exists",
            severity="INFO",
            passed=True,
            message=str(data_dir),
        )
    )

    splits_to_check = [target_split] if target_split else EXPECTED_SPLITS
    existing_splits: List[str] = []

    for split_name in splits_to_check:
        split_dir = data_dir / split_name
        if not split_dir.is_dir():
            report.add(
                CheckResult(
                    check_id=f"1.1_{split_name}",
                    category="1. Structural Integrity",
                    name=f"Split directory exists: {split_name}",
                    severity="ERROR",
                    passed=False,
                    message=f"Directory not found: {split_dir}",
                )
            )
            continue
        existing_splits.append(split_name)

        expected_subdirs = {"images_png"}
        if split_name != "test":
            expected_subdirs.add("masks_png")

        for subdir in expected_subdirs:
            sub_path = split_dir / subdir
            if sub_path.is_dir():
                report.add(
                    CheckResult(
                        check_id=f"1.1_{split_name}_{subdir}",
                        category="1. Structural Integrity",
                        name=f"Sub-directory exists: {split_name}/{subdir}",
                        severity="INFO",
                        passed=True,
                        message=str(sub_path),
                    )
                )
            else:
                report.add(
                    CheckResult(
                        check_id=f"1.1_{split_name}_{subdir}",
                        category="1. Structural Integrity",
                        name=f"Sub-directory exists: {split_name}/{subdir}",
                        severity="FATAL",
                        passed=False,
                        message=f"Missing: {sub_path}",
                    )
                )

        if split_name == "test":
            test_mask_dir = split_dir / "masks_png"
            if test_mask_dir.is_dir():
                report.add(
                    CheckResult(
                        check_id=f"1.1c_{split_name}",
                        category="1. Structural Integrity",
                        name=f"Test split has no masks_png folder",
                        severity="WARN",
                        passed=False,
                        message=f"Unexpected masks_png directory found in test",
                    )
                )
            else:
                report.add(
                    CheckResult(
                        check_id=f"1.1c_{split_name}",
                        category="1. Structural Integrity",
                        name=f"Test split has no masks_png folder",
                        severity="INFO",
                        passed=True,
                        message="OK",
                    )
                )

        manifest_path = split_dir / "manifest.csv"
        if not manifest_path.is_file():
            report.add(
                CheckResult(
                    check_id=f"1.2_{split_name}",
                    category="1. Structural Integrity",
                    name=f"Manifest exists: {split_name}",
                    severity="FATAL",
                    passed=False,
                    message=f"Missing: {manifest_path}",
                )
            )

        known_subdirs = {"images_png", "masks_png"}
        for item in split_dir.iterdir():
            if item.is_dir() and item.name not in known_subdirs:
                report.add(
                    CheckResult(
                        check_id=f"1.3_{split_name}_{item.name}",
                        category="1. Structural Integrity",
                        name=f"Unexpected sub-directory: {split_name}/{item.name}",
                        severity="WARN",
                        passed=False,
                        message=str(item),
                    )
                )

    if not existing_splits:
        report.add(
            CheckResult(
                check_id="1.x",
                category="1. Structural Integrity",
                name="At least one split directory found",
                severity="FATAL",
                passed=False,
                message="No split directories present",
            )
        )
        return report

    split_manifests: Dict[str, List[Dict[str, str]]] = {}
    split_stems: Dict[str, Set[str]] = {}
    split_tile_paths: Dict[str, Set[str]] = {}

    for split_name in existing_splits:
        manifest_path = data_dir / split_name / "manifest.csv"
        rows = _parse_manifest(manifest_path, report, f"1.2_{split_name}")
        split_manifests[split_name] = rows
        split_stems[split_name] = _get_split_stems(rows)
        split_tile_paths[split_name] = _get_tile_paths(rows)

    _TILE_MISSING_WARNED: Set[str] = set()

    for split_name in existing_splits:
        rows = split_manifests.get(split_name, [])
        split_dir = data_dir / split_name
        img_dir = split_dir / "images_png"
        mask_dir = split_dir / "masks_png" if split_name != "test" else None

        missing_tiles = 0
        total_tiles = 0

        for row in rows:
            tile_path_str = row.get("tile_path", "")
            mask_path_str = row.get("mask_path", "")

            if tile_path_str:
                total_tiles += 1
                tile_file = Path(tile_path_str)
                if not tile_file.is_file():
                    missing_tiles += 1
                    if tile_path_str not in _TILE_MISSING_WARNED:
                        report.add(
                            CheckResult(
                                check_id=f"2.1_{split_name}",
                                category="2. File Integrity",
                                name=f"Tile exists on disk: {tile_file.name}",
                                severity="FATAL",
                                passed=False,
                                message=f"Missing: {tile_path_str}",
                            )
                        )
                        _TILE_MISSING_WARNED.add(tile_path_str)

            if mask_path_str and mask_dir and mask_dir.is_dir():
                mask_file = Path(mask_path_str)
                if not mask_file.is_file():
                    if mask_path_str not in _TILE_MISSING_WARNED:
                        report.add(
                            CheckResult(
                                check_id=f"2.2_{split_name}",
                                category="2. File Integrity",
                                name=f"Mask exists on disk: {mask_file.name}",
                                severity="FATAL",
                                passed=False,
                                message=f"Missing: {mask_path_str}",
                            )
                        )
                        _TILE_MISSING_WARNED.add(mask_path_str)

        if missing_tiles == 0 and total_tiles > 0:
            report.add(
                CheckResult(
                    check_id=f"2.1_summary_{split_name}",
                    category="2. File Integrity",
                    name=f"All tile files on disk: {split_name}",
                    severity="INFO",
                    passed=True,
                    message=f"OK — {total_tiles} tiles verified",
                )
            )

        if img_dir.is_dir():
            png_files = sorted([f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"])
            manifest_count = len(rows)
            if len(png_files) != manifest_count:
                report.add(
                    CheckResult(
                        check_id=f"2.3_{split_name}",
                        category="2. File Integrity",
                        name=f"PNG count matches manifest rows: {split_name}",
                        severity="ERROR",
                        passed=False,
                        message=f"Disk: {len(png_files)} PNGs, Manifest: {manifest_count} rows",
                    )
                )
            else:
                report.add(
                    CheckResult(
                        check_id=f"2.3_{split_name}",
                        category="2. File Integrity",
                        name=f"PNG count matches manifest rows: {split_name}",
                        severity="INFO",
                        passed=True,
                        message=f"{len(png_files)} PNGs = {manifest_count} manifest rows",
                    )
                )

            manifest_tile_names = {Path(r.get("tile_path", "")).name for r in rows}
            disk_names = {f.name for f in png_files}
            unreferenced = disk_names - manifest_tile_names
            if unreferenced:
                report.add(
                    CheckResult(
                        check_id=f"2.5_{split_name}",
                        category="2. File Integrity",
                        name=f"No orphan PNGs on disk: {split_name}",
                        severity="WARN",
                        passed=False,
                        message=f"{len(unreferenced)} PNGs not in manifest",
                    )
                )
            else:
                report.add(
                    CheckResult(
                        check_id=f"2.5_{split_name}",
                        category="2. File Integrity",
                        name=f"No orphan PNGs on disk: {split_name}",
                        severity="INFO",
                        passed=True,
                        message="OK",
                    )
                )

    image_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for split_name in existing_splits:
        split_dir = data_dir / split_name
        img_dir = split_dir / "images_png"
        if not img_dir.is_dir():
            continue

        all_img_files = sorted(
            [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"]
        )

        if full_scan:
            sample_images = all_img_files
        elif sample_frac < 1.0:
            step = max(1, int(1.0 / sample_frac))
            sample_images = all_img_files[::step]
        else:
            sample_images = all_img_files

        dim_failures = 0
        mode_failures = 0
        corrupt_failures = 0
        black_images = 0
        white_images = 0
        total_imgs_checked = 0

        for img_file in sample_images:
            total_imgs_checked += 1
            filepath = img_file

            arr = _validate_image_file(
                filepath, report, f"3_{split_name}_{img_file.name}", "3. Image Validation"
            )
            if arr is None:
                corrupt_failures += 1
                continue

            h, w = arr.shape[:2]
            if (h, w) != EXPECTED_TILE_SIZE:
                dim_failures += 1
                if dim_failures <= 5:
                    report.add(
                        CheckResult(
                            check_id=f"3.1_{split_name}_{img_file.name}",
                            category="3. Image Validation",
                            name=f"Image dimensions = 512x512: {split_name}/{img_file.name}",
                            severity="FATAL",
                            passed=False,
                            message=f"Got {h}x{w}",
                        )
                    )

            if len(arr.shape) != 3 or arr.shape[2] != 3:
                mode_failures += 1
                if mode_failures <= 5:
                    report.add(
                        CheckResult(
                            check_id=f"3.2_{split_name}_{img_file.name}",
                            category="3. Image Validation",
                            name=f"Image is 3-channel RGB: {split_name}/{img_file.name}",
                            severity="FATAL",
                            passed=False,
                            message=f"Shape={arr.shape}",
                        )
                    )
                continue

            if arr.dtype != np.uint8:
                if mode_failures <= 5:
                    report.add(
                        CheckResult(
                            check_id=f"3.5_{split_name}_{img_file.name}",
                            category="3. Image Validation",
                            name=f"Image dtype is uint8: {split_name}/{img_file.name}",
                            severity="WARN",
                            passed=False,
                            message=f"dtype={arr.dtype}",
                        )
                    )

            for c in range(3):
                ch_mean = float(arr[:, :, c].mean())
                ch_std = float(arr[:, :, c].std())
                pmin = int(arr[:, :, c].min())
                pmax = int(arr[:, :, c].max())
                image_stats[split_name][f"ch{c}_mean"].append(ch_mean)
                image_stats[split_name][f"ch{c}_std"].append(ch_std)
                image_stats[split_name][f"ch{c}_min"].append(float(pmin))
                image_stats[split_name][f"ch{c}_max"].append(float(pmax))

            pmax_overall = int(arr.max())
            pmin_overall = int(arr.min())
            if pmax_overall == 0:
                black_images += 1
            if pmin_overall == 255:
                white_images += 1

            if split_name == "test":
                continue

            mask_dir = split_dir / "masks_png"
            mask_path = mask_dir / img_file.name
            mask_arr = _validate_image_file(
                mask_path, report, f"4_{split_name}_{img_file.name}", "4. Mask Validation"
            )
            if mask_arr is None:
                continue

            mh, mw = mask_arr.shape[:2]
            if (mh, mw) != EXPECTED_TILE_SIZE:
                report.add(
                    CheckResult(
                        check_id=f"4.1_{split_name}_{img_file.name}",
                        category="4. Mask Validation",
                        name=f"Mask dimensions = 512x512: {split_name}/{img_file.name}",
                        severity="FATAL",
                        passed=False,
                        message=f"Got {mh}x{mw}",
                    )
                )

            if len(mask_arr.shape) != 2:
                report.add(
                    CheckResult(
                        check_id=f"4.2_{split_name}_{img_file.name}",
                        category="4. Mask Validation",
                        name=f"Mask single-channel: {split_name}/{img_file.name}",
                        severity="FATAL",
                        passed=False,
                        message=f"Shape={mask_arr.shape}",
                    )
                )

            _validate_mask_values(mask_arr, mask_path, report, f"4_{split_name}_{img_file.name}")

            unique_mask = np.unique(mask_arr)
            if set(unique_mask.tolist()) == {0}:
                report.add(
                    CheckResult(
                        check_id=f"4.8_{split_name}_{img_file.name}",
                        category="4. Mask Validation",
                        name=f"All-ignore mask: {split_name}/{img_file.name}",
                        severity="INFO",
                        passed=True,
                        message="100% class 0 (ignore)",
                    )
                )
            if set(unique_mask.tolist()) == {1}:
                report.add(
                    CheckResult(
                        check_id=f"4.9_{split_name}_{img_file.name}",
                        category="4. Mask Validation",
                        name=f"All-background mask: {split_name}/{img_file.name}",
                        severity="INFO",
                        passed=True,
                        message="100% class 1 (background)",
                    )
                )

        if dim_failures:
            report.add(
                CheckResult(
                    check_id=f"3.1_summary_{split_name}",
                    category="3. Image Validation",
                    name=f"All tiles 512x512: {split_name}",
                    severity="FATAL",
                    passed=False,
                    message=f"{dim_failures}/{total_imgs_checked} tiles have wrong dimensions",
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id=f"3.1_summary_{split_name}",
                    category="3. Image Validation",
                    name=f"All tiles 512x512: {split_name}",
                    severity="INFO",
                    passed=True,
                    message=f"OK — {total_imgs_checked} checked",
                )
            )

        if mode_failures:
            report.add(
                CheckResult(
                    check_id=f"3.2_summary_{split_name}",
                    category="3. Image Validation",
                    name=f"All tiles 3-channel RGB: {split_name}",
                    severity="FATAL",
                    passed=False,
                    message=f"{mode_failures}/{total_imgs_checked} tiles not RGB",
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id=f"3.2_summary_{split_name}",
                    category="3. Image Validation",
                    name=f"All tiles 3-channel RGB: {split_name}",
                    severity="INFO",
                    passed=True,
                    message=f"OK — {total_imgs_checked} checked",
                )
            )

        if black_images:
            report.add(
                CheckResult(
                    check_id=f"3.7_summary_{split_name}",
                    category="3. Image Validation",
                    name=f"No all-black tiles: {split_name}",
                    severity="WARN",
                    passed=False,
                    message=f"{black_images} all-black tiles",
                )
            )

        if white_images:
            report.add(
                CheckResult(
                    check_id=f"3.8_summary_{split_name}",
                    category="3. Image Validation",
                    name=f"No all-white tiles: {split_name}",
                    severity="WARN",
                    passed=False,
                    message=f"{white_images} all-white tiles",
                )
            )

    for split_name in existing_splits:
        stats = image_stats[split_name]
        split_summary: Dict[str, Any] = {}
        for key, values in stats.items():
            if values:
                global_mean = float(np.mean(values))
                global_std = float(np.std(values))
                split_summary[key] = {"mean": round(global_mean, 2), "std": round(global_std, 2)}
        if split_summary:
            report.per_split_stats.setdefault(split_name, {}).update(split_summary)


    for split_name in existing_splits:
        if split_name == "test":
            continue
        split_dir = data_dir / split_name
        mask_dir = split_dir / "masks_png"
        if not mask_dir.is_dir():
            continue

        class_counts: Dict[int, int] = defaultdict(int)
        total_non_ignore = 0
        total_pixels = 0

        all_mask_files = sorted(
            [f for f in mask_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"]
        )

        if full_scan:
            sample_masks = all_mask_files
        elif sample_frac < 1.0:
            step = max(1, int(1.0 / sample_frac))
            sample_masks = all_mask_files[::step]
        else:
            sample_masks = all_mask_files

        for mf in sample_masks:
            try:
                img = Image.open(mask_dir / mf)
                arr = np.array(img)
                img.close()
                unique_vals, counts_np = np.unique(arr, return_counts=True)
                for v, c in zip(unique_vals.tolist(), counts_np.tolist()):
                    class_counts[v] += c
                    if v != 0:
                        total_non_ignore += c
                    total_pixels += c
            except Exception:
                continue

        if total_non_ignore > 0:
            report.class_distribution[split_name] = dict(class_counts)

            ignore_pct = (class_counts.get(0, 0) / total_pixels * 100) if total_pixels > 0 else 0.0
            if ignore_pct > 30:
                report.add(
                    CheckResult(
                        check_id=f"5.3_{split_name}",
                        category="5. Class Distribution",
                        name=f"Ignore ratio ≤ 30%: {split_name}",
                        severity="WARN",
                        passed=False,
                        message=f"Ignore = {ignore_pct:.1f}% of total pixels",
                    )
                )
            else:
                report.add(
                    CheckResult(
                        check_id=f"5.3_{split_name}",
                        category="5. Class Distribution",
                        name=f"Ignore ratio ≤ 30%: {split_name}",
                        severity="INFO",
                        passed=True,
                        message=f"Ignore = {ignore_pct:.1f}%",
                    )
                )

            for cls_id in sorted(class_counts.keys()):
                if cls_id == 0:
                    continue
                pct = (class_counts[cls_id] / total_non_ignore * 100) if total_non_ignore > 0 else 0.0
                if pct < 0.1:
                    report.add(
                        CheckResult(
                            check_id=f"5.5_{split_name}_cls{cls_id}",
                            category="5. Class Distribution",
                            name=f"Class {cls_id} critically rare (<0.1%): {split_name}",
                            severity="ERROR",
                            passed=False,
                            message=f"{LOVEDA_CLASS_NAMES.get(cls_id, cls_id)}: {pct:.3f}%",
                        )
                    )
                elif pct < 1.0:
                    report.add(
                        CheckResult(
                            check_id=f"5.4_{split_name}_cls{cls_id}",
                            category="5. Class Distribution",
                            name=f"Class {cls_id} rare (<1%): {split_name}",
                            severity="WARN",
                            passed=False,
                            message=f"{LOVEDA_CLASS_NAMES.get(cls_id, cls_id)}: {pct:.2f}%",
                        )
                    )

            present_non_ignore = set(class_counts.keys()) & NON_IGNORE_CLASSES
            missing_classes = NON_IGNORE_CLASSES - present_non_ignore
            if missing_classes:
                report.add(
                    CheckResult(
                        check_id=f"5.6_{split_name}",
                        category="5. Class Distribution",
                        name=f"All 7 non-ignore classes present: {split_name}",
                        severity="ERROR",
                        passed=False,
                        message=f"Missing class(es): {sorted(missing_classes)}",
                    )
                )
            else:
                report.add(
                    CheckResult(
                        check_id=f"5.6_{split_name}",
                        category="5. Class Distribution",
                        name=f"All 7 non-ignore classes present: {split_name}",
                        severity="INFO",
                        passed=True,
                        message="OK",
                    )
                )


    train_stems = split_stems.get("train", set())
    val_stems = split_stems.get("val", set())
    test_stems = split_stems.get("test", set())

    train_val_overlap = train_stems & val_stems
    if train_val_overlap:
        report.add(
            CheckResult(
                check_id="6.1",
                category="6. Split Integrity",
                name="No train ↔ val source stem leakage",
                severity="FATAL",
                passed=False,
                message=f"{len(train_val_overlap)} stems shared",
                details={"count": len(train_val_overlap), "examples": sorted(list(train_val_overlap))[:10]},
            )
        )
    else:
        report.add(
            CheckResult(
                check_id="6.1",
                category="6. Split Integrity",
                name="No train ↔ val source stem leakage",
                severity="INFO",
                passed=True,
                message="OK",
            )
        )

    train_test_overlap = train_stems & test_stems
    if train_test_overlap:
        report.add(
            CheckResult(
                check_id="6.2",
                category="6. Split Integrity",
                name="No train ↔ test source stem leakage",
                severity="FATAL",
                passed=False,
                message=f"{len(train_test_overlap)} stems shared",
                details={"count": len(train_test_overlap), "examples": sorted(list(train_test_overlap))[:10]},
            )
        )
    else:
        report.add(
            CheckResult(
                check_id="6.2",
                category="6. Split Integrity",
                name="No train ↔ test source stem leakage",
                severity="INFO",
                passed=True,
                message="OK",
            )
        )

    val_test_overlap = val_stems & test_stems
    if val_test_overlap:
        report.add(
            CheckResult(
                check_id="6.3",
                category="6. Split Integrity",
                name="No val ↔ test source stem leakage",
                severity="FATAL",
                passed=False,
                message=f"{len(val_test_overlap)} stems shared",
                details={"count": len(val_test_overlap), "examples": sorted(list(val_test_overlap))[:10]},
            )
        )
    else:
        report.add(
            CheckResult(
                check_id="6.3",
                category="6. Split Integrity",
                name="No val ↔ test source stem leakage",
                severity="INFO",
                passed=True,
                message="OK",
            )
        )

    for split_name in existing_splits:
        rows = split_manifests.get(split_name, [])
        stem_to_tiles: Dict[str, List[str]] = defaultdict(list)
        for row in rows:
            stem = row.get("source_stem", "")
            tile = row.get("tile_path", "")
            if stem and tile:
                stem_to_tiles[stem].append(tile)
        wrong_count = sum(1 for tiles in stem_to_tiles.values() if len(tiles) != EXPECTED_TILES_PER_SOURCE)
        if wrong_count:
            report.add(
                CheckResult(
                    check_id=f"6.4_{split_name}",
                    category="6. Split Integrity",
                    name=f"Each source has exactly 4 tiles: {split_name}",
                    severity="ERROR",
                    passed=False,
                    message=f"{wrong_count} sources have ≠ 4 tiles",
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id=f"6.4_{split_name}",
                    category="6. Split Integrity",
                    name=f"Each source has exactly 4 tiles: {split_name}",
                    severity="INFO",
                    passed=True,
                    message=f"OK — {len(stem_to_tiles)} sources × 4 tiles",
                )
            )

    for split_name in existing_splits:
        rows = split_manifests.get(split_name, [])
        tile_paths_list = [r.get("tile_path", "") for r in rows if r.get("tile_path", "")]
        gids = [r.get("global_id", "") for r in rows if r.get("global_id", "")]

        dup_tiles = len(tile_paths_list) != len(set(tile_paths_list))
        dup_gids = len(gids) != len(set(gids))

        if dup_tiles:
            report.add(
                CheckResult(
                    check_id=f"6.6_{split_name}",
                    category="6. Split Integrity",
                    name=f"No duplicate tile_paths: {split_name}",
                    severity="ERROR",
                    passed=False,
                    message="Duplicate tile_paths detected",
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id=f"6.6_{split_name}",
                    category="6. Split Integrity",
                    name=f"No duplicate tile_paths: {split_name}",
                    severity="INFO",
                    passed=True,
                    message="OK",
                )
            )
        if dup_gids:
            report.add(
                CheckResult(
                    check_id=f"6.7_{split_name}",
                    category="6. Split Integrity",
                    name=f"No duplicate global_ids: {split_name}",
                    severity="ERROR",
                    passed=False,
                    message="Duplicate global_ids detected",
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id=f"6.7_{split_name}",
                    category="6. Split Integrity",
                    name=f"No duplicate global_ids: {split_name}",
                    severity="INFO",
                    passed=True,
                    message="OK",
                )
            )

    for split_name in existing_splits:
        split_dir = data_dir / split_name
        img_dir = split_dir / "images_png"
        if not img_dir.is_dir():
            continue
        png_files = sorted(
            [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"],
            key=lambda f: int(f.stem) if f.stem.isdigit() else -1,
        )
        expected_names = [f"{i}.png" for i in range(len(png_files))]
        mismatches = [
            (pf.name, en)
            for pf, en in zip(png_files, expected_names)
            if pf.name != en
        ]
        if mismatches:
            report.add(
                CheckResult(
                    check_id=f"6.8_{split_name}",
                    category="6. Split Integrity",
                    name=f"Tiles renamed sequentially (0,1,2,…): {split_name}",
                    severity="WARN",
                    passed=False,
                    message=f"{len(mismatches)} naming gaps/mismatches, e.g. {mismatches[:3]}",
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id=f"6.8_{split_name}",
                    category="6. Split Integrity",
                    name=f"Tiles renamed sequentially (0,1,2,…): {split_name}",
                    severity="INFO",
                    passed=True,
                    message=f"OK — 0..{len(png_files) - 1} contiguous",
                )
            )


    if "test" in existing_splits:
        test_rows = split_manifests.get("test", [])

        non_empty_masks = [r for r in test_rows if r.get("mask_path", "").strip()]
        if non_empty_masks:
            report.add(
                CheckResult(
                    check_id="7.1",
                    category="7. Test Set Validation",
                    name="Test manifest mask_path is empty",
                    severity="WARN",
                    passed=False,
                    message=f"{len(non_empty_masks)} rows have non-empty mask_path",
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id="7.1",
                    category="7. Test Set Validation",
                    name="Test manifest mask_path is empty",
                    severity="INFO",
                    passed=True,
                    message="OK",
                )
            )

        not_test_orig = [r for r in test_rows if r.get("orig_split", "").strip() != "Test"]
        if not_test_orig:
            report.add(
                CheckResult(
                    check_id="7.2",
                    category="7. Test Set Validation",
                    name="Test sources marked orig_split='Test'",
                    severity="WARN",
                    passed=False,
                    message=f"{len(not_test_orig)} rows have unexpected orig_split",
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id="7.2",
                    category="7. Test Set Validation",
                    name="Test sources marked orig_split='Test'",
                    severity="INFO",
                    passed=True,
                    message="OK",
                )
            )

        not_test_assigned = [r for r in test_rows if r.get("assigned_split", "").strip() != "test"]
        if not_test_assigned:
            report.add(
                CheckResult(
                    check_id="7.3",
                    category="7. Test Set Validation",
                    name="Test sources marked assigned_split='test'",
                    severity="WARN",
                    passed=False,
                    message=f"{len(not_test_assigned)} rows have unexpected assigned_split",
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id="7.3",
                    category="7. Test Set Validation",
                    name="Test sources marked assigned_split='test'",
                    severity="INFO",
                    passed=True,
                    message="OK",
                )
            )


    for split_name in existing_splits:
        rows = split_manifests.get(split_name, [])
        if not rows:
            continue

        urban_rows = [r for r in rows if r.get("region", "").lower().startswith("urban")]
        rural_rows = [r for r in rows if r.get("region", "").lower().startswith("rural")]
        other_rows = [r for r in rows if r.get("region", "").lower() not in ("urban", "rural")]
        total = len(rows)

        urban_src_count = len({r.get("source_stem", "") for r in urban_rows})
        rural_src_count = len({r.get("source_stem", "") for r in rural_rows})

        if total > 0:
            urban_pct = len(urban_rows) / total * 100

            if split_name == "train":
                train_urban_src_pct = urban_src_count / (urban_src_count + rural_src_count) * 100 if (urban_src_count + rural_src_count) > 0 else 0
                if abs(len(urban_rows) / total - EXPECTED_TRAIN_URBAN_FRAC) > 0.03:
                    report.add(
                        CheckResult(
                            check_id="8.1_train",
                            category="8. Urban/Rural Balance",
                            name=f"Train Urban fraction ≈ {EXPECTED_TRAIN_URBAN_FRAC:.0%}",
                            severity="WARN",
                            passed=False,
                            message=(
                                f"Expected ~{EXPECTED_TRAIN_URBAN_FRAC:.0%}, "
                                f"got {len(urban_rows)}/{total} = {urban_pct:.1f}% tiles "
                                f"({urban_src_count} Urban / {rural_src_count} Rural sources)"
                            ),
                        )
                    )
                else:
                    report.add(
                        CheckResult(
                            check_id="8.1_train",
                            category="8. Urban/Rural Balance",
                            name=f"Train Urban fraction ≈ {EXPECTED_TRAIN_URBAN_FRAC:.0%}",
                            severity="INFO",
                            passed=True,
                            message=(
                                f"{len(urban_rows)} Urban / {len(rural_rows)} Rural tiles "
                                f"= {urban_pct:.1f}% "
                                f"({urban_src_count} Urban / {rural_src_count} Rural sources)"
                            ),
                        )
                    )

            elif split_name == "val":
                if abs(urban_src_count - rural_src_count) > 5:
                    report.add(
                        CheckResult(
                            check_id="8.2_val",
                            category="8. Urban/Rural Balance",
                            name="Validation Urban/Rural ≈ 50/50 by source count",
                            severity="WARN",
                            passed=False,
                            message=(
                                f"{urban_src_count} Urban / {rural_src_count} Rural sources — "
                                f"diff={abs(urban_src_count - rural_src_count)}"
                            ),
                        )
                    )
                else:
                    report.add(
                        CheckResult(
                            check_id="8.2_val",
                            category="8. Urban/Rural Balance",
                            name="Validation Urban/Rural ≈ 50/50 by source count",
                            severity="INFO",
                            passed=True,
                            message=f"{urban_src_count} Urban / {rural_src_count} Rural sources (balanced)",
                        )
                    )

            if other_rows:
                report.add(
                    CheckResult(
                        check_id=f"8.3_{split_name}",
                        category="8. Urban/Rural Balance",
                        name=f"No unknown region values: {split_name}",
                        severity="WARN",
                        passed=False,
                        message=f"{len(other_rows)} rows with non-standard region value",
                    )
                )
            else:
                report.add(
                    CheckResult(
                        check_id=f"8.3_{split_name}",
                        category="8. Urban/Rural Balance",
                        name=f"All region values are Urban or Rural: {split_name}",
                        severity="INFO",
                        passed=True,
                        message="OK",
                    )
                )


    total_bytes = 0
    for split_name in existing_splits:
        split_dir = data_dir / split_name
        split_bytes = 0
        tile_count = 0
        for f in split_dir.rglob("*.png"):
            if f.is_file():
                split_bytes += f.stat().st_size
                tile_count += 1
        total_bytes += split_bytes
        split_gb = split_bytes / (1024 ** 3)
        report.per_split_stats.setdefault(split_name, {}).update(
            {"disk_size_gb": round(split_gb, 3), "png_file_count": tile_count}
        )
        report.add(
            CheckResult(
                check_id=f"9.2_{split_name}",
                category="9. Disk Footprint",
                name=f"Split disk size: {split_name}",
                severity="INFO",
                passed=True,
                message=f"{split_gb:.2f} GB, {tile_count} PNG files",
            )
        )

    total_gb = total_bytes / (1024 ** 3)
    report.add(
        CheckResult(
            check_id="9.1",
            category="9. Disk Footprint",
            name="Total dataset size",
            severity="INFO",
            passed=True,
            message=f"{total_gb:.2f} GB",
            details={"total_gb": round(total_gb, 3)},
        )
    )

    for split_name in existing_splits:
        rows = split_manifests.get(split_name, [])
        tile_count = len(rows)
        report.per_split_stats.setdefault(split_name, {}).update({"manifest_tile_count": tile_count})
        if split_name == "train":
            samples_per_epoch = tile_count // batch_size if batch_size > 0 else 0
            report.add(
                CheckResult(
                    check_id="9.4",
                    category="9. Disk Footprint",
                    name="Samples per epoch (train)",
                    severity="INFO",
                    passed=True,
                    message=f"{samples_per_epoch} steps/epoch at batch_size={batch_size}",
                )
            )

    gpu_est = round(1.2 + 0.25 * batch_size, 1)
    report.add(
        CheckResult(
            check_id="9.5",
            category="9. Disk Footprint",
            name="GPU memory estimate (U-Net++ ResNeXt-50 @ 512×512)",
            severity="INFO",
            passed=True,
            message=f"~{gpu_est} GB at batch_size={batch_size}",
        )
    )


    for split_name in existing_splits:
        rows = split_manifests.get(split_name, [])
        if not rows:
            continue
        source_tile_count: Dict[str, int] = defaultdict(int)
        for row in rows:
            stem = row.get("source_stem", "")
            if stem:
                source_tile_count[stem] += 1

        incomplete_sources = sum(1 for c in source_tile_count.values() if c != EXPECTED_TILES_PER_SOURCE)
        extra_sources = sum(1 for c in source_tile_count.values() if c > EXPECTED_TILES_PER_SOURCE)
        missing_sources = sum(1 for c in source_tile_count.values() if c < EXPECTED_TILES_PER_SOURCE)

        if incomplete_sources:
            report.add(
                CheckResult(
                    check_id=f"10.1_{split_name}",
                    category="10. Source Completeness",
                    name=f"All sources have exactly 4 tiles: {split_name}",
                    severity="ERROR",
                    passed=False,
                    message=(
                        f"{incomplete_sources} sources with ≠4 tiles "
                        f"({missing_sources} <4, {extra_sources} >4) "
                        f"out of {len(source_tile_count)} sources"
                    ),
                    details={
                        "total_sources": len(source_tile_count),
                        "incomplete": incomplete_sources,
                        "too_few": missing_sources,
                        "too_many": extra_sources,
                    },
                )
            )
        else:
            report.add(
                CheckResult(
                    check_id=f"10.1_{split_name}",
                    category="10. Source Completeness",
                    name=f"All sources have exactly 4 tiles: {split_name}",
                    severity="INFO",
                    passed=True,
                    message=f"{len(source_tile_count)} sources × 4 tiles OK",
                )
            )

    return report




def main() -> None:
    if sys.stdout.encoding.lower() in ("cp1252", "cp850", "cp437"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Urbaneuron — 03_data_quality.py: Validate data/final/ before training. (v13)"
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Validate every single tile (default: sampled scan).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Validate a single split only (e.g., 'train', 'val', 'test').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for GPU memory estimate (default: 8).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.10,
        help="Fraction of tiles to sample when not in full-scan (default: 0.10).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/final"),
        help="Path to final data directory (default: data/final).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports (default: same as --data-dir).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON only, suppress text report to stdout.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir or data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  URBANEURON — 03_data_quality.py  (v13)")
    print("=" * 72)
    print(f"  Data directory : {data_dir}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Scan mode      : {'Full scan' if args.full_scan else f'Sampled (~{args.sample_frac:.0%})'}")
    print(f"  Target split   : {args.split or 'ALL'}")
    print(f"  Batch size     : {args.batch_size}")
    print("=" * 72)
    print()

    report = run_validation(
        data_dir=data_dir,
        full_scan=args.full_scan,
        target_split=args.split,
        batch_size=args.batch_size,
        sample_frac=min(1.0, args.sample_frac),
        output_dir=output_dir,
    )

    text_report = report.text_summary()
    txt_path = output_dir / "validation_report.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text_report)
    if not args.json_only:
        print(text_report)
    print(f"\n[INFO] Text report saved to: {txt_path}")

    json_report = report.json_summary()
    json_path = output_dir / "validation_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json_report, fh, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON report saved to: {json_path}")

    if report.all_passed():
        print("\n[PASS] All critical checks succeeded.")
        sys.exit(0)
    else:
        print("\n[FAIL] One or more FATAL/ERROR checks failed. See report for details.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
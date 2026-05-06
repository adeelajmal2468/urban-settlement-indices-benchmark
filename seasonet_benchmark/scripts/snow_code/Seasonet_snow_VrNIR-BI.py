#!/usr/bin/env python
"""
Compute VrNIR BI urban index on SeasoNet snow dataset and evaluate
urban vs non urban segmentation against pixel labels.

Assumptions
----------
* Data root: D:\\SeasoNet
* CSV: snow_entries_with_labels.csv
  must contain at least column: Path
  (relative path starting with "snow/...")
* Every tile folder contains:
    *_10m_RGB.tif      (3 bands, R G B, 120 x 120)
    *_10m_IR.tif       (1 band, NIR, 120 x 120)
    *_labels.tif       (1 band, classes in [0..33])
    *_clouds.tif       (1 band, 0 or 1)
    *_snow.tif         (1 band, 0 or 1)

Outputs
-------
* D:\\SeasoNet\\vrnir_bi_results\\global_metrics.json
* D:\\SeasoNet\\vrnir_bi_results\\confusion_matrix.png
* D:\\SeasoNet\\vrnir_bi_results\\overlays\\<tile_id>_overlay.png
"""

import os
import glob
import json
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from skimage.filters import threshold_otsu
import matplotlib

# non interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------

DATA_ROOT = r"D:\SeasoNet"
CSV_PATH = r"D:\SeasoNet\snow_entries_with_labels.csv"
OUTPUT_ROOT = r"D:\SeasoNet\vrnir_bi_results"

# process only this many tiles first (set to None to process all)
MAX_TILES = 100

# limit number of overlay figures if desired (None = one per processed tile)
MAX_OVERLAYS = 100  # can also set to None for all overlays

# classes 1..11 are urban (change here if you want a different mapping)
URBAN_CLASSES = set(range(1, 12))
ALL_CLASSES = set(range(1, 34))  # 1..33

os.makedirs(OUTPUT_ROOT, exist_ok=True)
OVERLAY_DIR = os.path.join(OUTPUT_ROOT, "overlays")
os.makedirs(OVERLAY_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------

def find_single_file(folder, patterns):
    """
    Search for a single file in `folder` that matches one of the patterns.
    Returns the path or None.
    """
    for pat in patterns:
        matches = glob.glob(os.path.join(folder, pat))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            matches_sorted = sorted(matches, key=len)
            return matches_sorted[0]
    return None


def load_tiff(path):
    """Load a GeoTIFF into a numpy array, return array and profile."""
    with rasterio.open(path) as src:
        arr = src.read()  # shape (bands, H, W)
        profile = src.profile
    return arr, profile


def compute_vrnir_bi(red, nir, eps=1e-6):
    """
    VrNIR BI = (Red - NIR) / (Red + NIR)

    red and nir are numpy arrays of same shape.
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    return (red - nir) / (red + nir + eps)


def labels_to_binary(labels):
    """
    Convert multi class labels to binary urban vs non urban.

    labels: numpy array with integer classes.
    Returns:
        binary (uint8) with 1 for urban, 0 for non urban
        valid_mask (bool) True where label is in known classes
    """
    labels = labels.astype(np.int32)
    valid_mask = np.isin(labels, list(ALL_CLASSES))
    urban_mask = np.isin(labels, list(URBAN_CLASSES))
    binary = np.zeros_like(labels, dtype=np.uint8)
    binary[urban_mask] = 1
    return binary, valid_mask


def update_confusion(y_true, y_pred):
    """
    Update global confusion counts.

    y_true, y_pred: 1D arrays of 0 or 1 of same length.
    Returns tp, fp, fn, tn.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    tn = np.logical_and(~y_true, ~y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()
    return int(tp), int(fp), int(fn), int(tn)


def metrics_from_confusion(tp, fp, fn, tn, eps=1e-9):
    total = tp + fp + fn + tn
    acc = (tp + tn) / (total + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou_urban = tp / (tp + fp + fn + eps)
    iou_non = tn / (tn + fp + fn + eps)
    miou = 0.5 * (iou_urban + iou_non)
    po = acc
    pe = (
        ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn))
        / ((total + eps) ** 2)
    )
    kappa = (po - pe) / (1 - pe + eps)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou_urban": float(iou_urban),
        "iou_non": float(iou_non),
        "miou": float(miou),
        "kappa": float(kappa),
    }


def save_overlay(tile_id, index_map, gt_binary, out_path):
    """
    Save overlay figure with three panels:
      1. VrNIR BI index (gray)
      2. binary ground truth (0/1, black/white)
      3. index in gray with colored overlay (urban vs non urban)
    """
    index_2d = np.squeeze(index_map)
    gt_2d = np.squeeze(gt_binary)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: index in gray
    im0 = axes[0].imshow(index_2d, cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title("VrNIR BI (gray)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: binary label (black/white)
    cmap_bin = ListedColormap(["black", "white"])
    axes[1].imshow(gt_2d, cmap=cmap_bin, vmin=0, vmax=1)
    axes[1].set_title("Binary label\n0 = non urban, 1 = urban")
    axes[1].axis("off")

    # Panel 3: gray index + colored overlay
    axes[2].imshow(index_2d, cmap="gray", vmin=-1, vmax=1)
    overlay = np.zeros((gt_2d.shape[0], gt_2d.shape[1], 4), dtype=np.float32)
    # blue for non urban, red for urban
    overlay[gt_2d == 0] = [0, 0, 1, 0.35]
    overlay[gt_2d == 1] = [1, 0, 0, 0.35]
    axes[2].imshow(overlay)
    axes[2].set_title("Index + overlay\nblue = non urban, red = urban")
    axes[2].axis("off")

    fig.suptitle(tile_id, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(tp, fp, fn, tn, out_path):
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=np.int64)

    classes = ["Non urban", "Urban"]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_ylabel("Ground truth")
    ax.set_xlabel("Prediction")
    ax.set_title("Confusion matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------

def process_dataset():
    df = pd.read_csv(CSV_PATH)

    if MAX_TILES is not None:
        df = df.head(MAX_TILES)

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_tn = 0

    overlay_counter = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing tiles"):
        rel_path = str(row["Path"])
        tile_dir = os.path.normpath(os.path.join(DATA_ROOT, rel_path))

        if not os.path.isdir(tile_dir):
            warnings.warn(f"Tile folder not found: {tile_dir}")
            continue

        rgb_path = find_single_file(tile_dir, ["*_10m_RGB.tif", "*_10m_RGB.tiff"])
        nir_path = find_single_file(tile_dir, ["*_10m_IR.tif", "*_10m_IR.tiff"])
        labels_path = find_single_file(tile_dir, ["*_labels.tif", "*_labels.tiff"])
        clouds_path = find_single_file(tile_dir, ["*_clouds.tif", "*_clouds.tiff"])
        snow_path = find_single_file(tile_dir, ["*_snow.tif", "*_snow.tiff"])

        if not all([rgb_path, nir_path, labels_path, clouds_path, snow_path]):
            warnings.warn(f"Missing files in {tile_dir}")
            continue

        try:
            rgb, _ = load_tiff(rgb_path)
            nir, _ = load_tiff(nir_path)
            labels, _ = load_tiff(labels_path)
            clouds, _ = load_tiff(clouds_path)
            snow, _ = load_tiff(snow_path)
        except RasterioIOError as e:
            warnings.warn(f"Raster read error in {tile_dir}: {e}")
            continue

        # rgb shape (3, H, W). red is band index 0 here
        red = rgb[0]
        nir_band = nir[0]

        index_map = compute_vrnir_bi(red, nir_band)

        gt_binary, valid_label_mask = labels_to_binary(labels[0])

        # mask clouds and snow (assume 1 means cloud or snow present)
        valid_mask = valid_label_mask & (clouds[0] == 0) & (snow[0] == 0)

        if not np.any(valid_mask):
            continue

        idx_values = index_map[valid_mask]

        try:
            thr = threshold_otsu(idx_values)
        except ValueError:
            thr = 0.0

        pred_binary = np.zeros_like(index_map, dtype=np.uint8)
        pred_binary[index_map >= thr] = 1

        y_true = gt_binary[valid_mask].ravel()
        y_pred = pred_binary[valid_mask].ravel()

        tp, fp, fn, tn = update_confusion(y_true, y_pred)
        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_tn += tn

        if MAX_OVERLAYS is None or overlay_counter < MAX_OVERLAYS:
            tile_id = os.path.basename(tile_dir.rstrip(os.sep))
            out_overlay = os.path.join(OVERLAY_DIR, f"{tile_id}_overlay.png")
            try:
                save_overlay(tile_id, index_map, gt_binary, out_overlay)
                overlay_counter += 1
            except Exception as e:
                warnings.warn(f"Overlay failed for {tile_dir}: {e}")

    metrics = metrics_from_confusion(global_tp, global_fp, global_fn, global_tn)

    metrics_path = os.path.join(OUTPUT_ROOT, "global_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    cm_path = os.path.join(OUTPUT_ROOT, "confusion_matrix.png")
    save_confusion_matrix(global_tp, global_fp, global_fn, global_tn, cm_path)

    print("Finished processing")
    print("Global metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    process_dataset()

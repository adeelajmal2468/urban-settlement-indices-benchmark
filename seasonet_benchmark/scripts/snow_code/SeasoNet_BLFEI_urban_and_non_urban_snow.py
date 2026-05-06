import os
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.enums import Resampling
from rasterio import warp
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, median_filter
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import threshold_otsu
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Define urban and non-urban classes and their colors
# -----------------------------------------------------------------------------
urban_classes = {
    1: "Continuous urban fabric",
    2: "Discontinuous urban fabric",
    3: "Industrial or commercial units",
    4: "Road and rail networks",
    5: "Port areas",
    6: "Airports",
    7: "Mineral extraction sites",
    8: "Dump sites",
    9: "Construction sites",
    10: "Green urban areas",
    11: "Sport and leisure facilities",
}
nonurban_classes = {
    12: "Non-irrigated arable land",
    13: "Vineyards",
    14: "Fruit trees and berry plantations",
    15: "Pastures",
    16: "Broad-leaved forest",
    17: "Coniferous forest",
    18: "Mixed forest",
    19: "Natural grasslands",
    20: "Moors and heathland",
    21: "Transitional woodland/shrub",
    22: "Beaches, dunes, sands",
    23: "Bare rock",
    24: "Sparsely vegetated areas",
    25: "Inland marshes",
    26: "Peat bogs",
    27: "Salt marshes",
    28: "Intertidal flats",
    29: "Water courses",
    30: "Water bodies",
    31: "Coastal lagoons",
    32: "Estuaries",
    33: "Sea and ocean",
}

urban_colors = {
    1: [1, 0, 0, 0.6],
    2: [0, 1, 0, 0.6],
    3: [0, 0, 1, 0.6],
    4: [1, 1, 0, 0.6],
    5: [1, 0, 1, 0.6],
    6: [0, 1, 1, 0.6],
    7: [0.8, 0.8, 0.8, 0.6],
    8: [0.7, 0.7, 0.7, 0.6],
    9: [0.6, 0.6, 0.6, 0.6],
    10: [0.5, 0.5, 0.5, 0.6],
    11: [0.5, 0, 0, 0.6],
}

nonurban_colors = {
    12: [0.5, 0.5, 0.0, 0.6],
    13: [0.0, 0.5, 0.0, 0.6],
    14: [0.0, 0.6, 0.0, 0.6],
    15: [0.0, 0.7, 0.0, 0.6],
    16: [0.0, 0.8, 0.0, 0.6],
    17: [0.0, 0.9, 0.0, 0.6],
    18: [0.0, 1.0, 0.0, 0.6],
    19: [0.1, 0.1, 0.1, 0.6],
    20: [0.2, 0.2, 0.2, 0.6],
    21: [0.3, 0.3, 0.3, 0.6],
    22: [0.4, 0.4, 0.4, 0.6],
    23: [0.5, 0.5, 0.5, 0.6],
    24: [0.6, 0.6, 0.6, 0.6],
    25: [0.7, 0.7, 0.7, 0.6],
    26: [0.8, 0.8, 0.8, 0.6],
    27: [0.9, 0.9, 0.9, 0.6],
    28: [1.0, 1.0, 1.0, 0.6],
    29: [0.0, 0.0, 0.5, 0.6],
    30: [0.0, 0.0, 1.0, 0.6],
    31: [0.0, 0.5, 1.0, 0.6],
    32: [0.0, 1.0, 1.0, 0.6],
    33: [0.0, 0.0, 0.8, 0.6],
}

# -----------------------------------------------------------------------------
# Helper Functions for Data Processing and Visualization
# -----------------------------------------------------------------------------
def load_and_preprocess_band(file_path, band_index):
    with rasterio.open(file_path) as src:
        band = src.read(band_index).astype(np.float32)
        band[band == src.nodata] = np.nan
        return band, src.meta


def resample_to_10m(source_path, band_index, target_meta):
    with rasterio.open(source_path) as src:
        band = src.read(band_index).astype(np.float32)
        resampled = np.empty((target_meta["height"], target_meta["width"]), dtype=np.float32)
        warp.reproject(
            source=band,
            destination=resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_meta["transform"],
            dst_crs=target_meta["crs"],
            resampling=Resampling.bilinear,
        )
    return resampled


# Corrected BLFEI function based on the paper:
def calculate_blfei(green, red, swir2, swir1):
    epsilon = 1e-6
    numerator = (green + red + swir2 - 3 * swir1) / 3
    denominator = (green + red + swir2 + 3 * swir1) / 3
    return numerator / (denominator + epsilon)


# Function to create a water mask using MNDWI
def create_water_mask(green, swir1, threshold=0.0):
    epsilon = 1e-6
    mndwi = (green - swir1) / (green + swir1 + epsilon)
    # Water pixels have MNDWI > threshold
    water_mask = mndwi > threshold
    return water_mask


def create_overlay(blfei, labels, threshold, urban_classes_present, nonurban_classes_present):
    overlay = np.zeros((*blfei.shape, 4))
    urban_mask = blfei > threshold
    for c in urban_classes_present:
        if c in urban_colors:
            class_mask = (labels == c) & urban_mask
            overlay[class_mask] = urban_colors[c]
    for c in nonurban_classes_present:
        if c in nonurban_colors:
            class_mask = (labels == c) & ~urban_mask
            overlay[class_mask] = nonurban_colors[c]
    return overlay


def plot_confusion_matrix(y_true, y_pred, title, class_names=None, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    default_class_names = ["Non-Urban", "Urban"]
    if class_names is None:
        class_names = default_class_names
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title, pad=20, fontsize=14)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.2,
                i + 0.1,
                f"({cm_percent[i, j]:.1f}%)",
                color="black",
                fontsize=10,
            )
    ax.figure.tight_layout()
    return cm


# -----------------------------------------------------------------------------
# Set Paths
# -----------------------------------------------------------------------------
csv_path = r"H:\filtered_snow_meta_with_both_urban_nonurban_2.csv"
base_path = r"H:\new_new_snow"
output_base = r"H:\new_new_snow\Indices Implementation\snow\blfei"

os.makedirs(output_base, exist_ok=True)


# -----------------------------------------------------------------------------
# Per-sample processing (called in worker processes)
# -----------------------------------------------------------------------------
def process_sample(i, folder_path_csv):
    folder_name = os.path.basename(folder_path_csv)
    folder_path = os.path.join(base_path, folder_path_csv)

    nir_tif_path = os.path.join(folder_path, f"{folder_name}_10m_IR.tif")
    rgb_tif_path = os.path.join(folder_path, f"{folder_name}_10m_RGB.tif")
    tif_20m_path = os.path.join(folder_path, f"{folder_name}_20m.tif")
    label_tif_path = os.path.join(folder_path, f"{folder_name}_labels.tif")

    # Load additional mask files
    easy_mask_path = os.path.join(folder_path, f"{folder_name}_mask_easy.tif")
    medium_mask_path = os.path.join(folder_path, f"{folder_name}_mask_medium.tif")
    clouds_path = os.path.join(folder_path, f"{folder_name}_clouds.tif")
    snow_path = os.path.join(folder_path, f"{folder_name}_snow.tif")
    if not (
        os.path.exists(easy_mask_path)
        and os.path.exists(medium_mask_path)
        and os.path.exists(clouds_path)
        and os.path.exists(snow_path)
    ):
        print(f"Skipping sample {folder_name}: additional mask files not found.")
        return None

    try:
        band_nir, meta_nir = load_and_preprocess_band(nir_tif_path, 1)
        band_red, _ = load_and_preprocess_band(rgb_tif_path, 3)  # Red band
        band_green, _ = load_and_preprocess_band(rgb_tif_path, 2)  # Green band
        band_swir_1 = resample_to_10m(tif_20m_path, 5, meta_nir)
        band_swir_2 = resample_to_10m(tif_20m_path, 6, meta_nir)
        labels, _ = load_and_preprocess_band(label_tif_path, 1)
        # Load the additional masks
        easy_mask, _ = load_and_preprocess_band(easy_mask_path, 1)
        medium_mask, _ = load_and_preprocess_band(medium_mask_path, 1)
        clouds_mask, _ = load_and_preprocess_band(clouds_path, 1)
        snow_mask, _ = load_and_preprocess_band(snow_path, 1)
    except Exception as e:
        print(f"Error loading data for {folder_name}: {e}")
        return None

    # Create a combined segmentation mask and a valid mask (excluding clouds and snow)
    seg_mask = (easy_mask > 0) | (medium_mask > 0)
    valid_mask = seg_mask & (clouds_mask == 0) & (snow_mask == 0)

    # Create water mask using MNDWI and update valid_mask (exclude water pixels)
    water_mask = create_water_mask(band_green, band_swir_1, threshold=0.0)
    valid_mask = valid_mask & (~water_mask)

    # Calculate BLFEI using the corrected function
    blfei = calculate_blfei(band_green, band_red, band_swir_2, band_swir_1)
    blfei_smooth = gaussian_filter(blfei, sigma=5)

    # Values actually used for Otsu (finite & valid mask)
    valid_for_otsu = blfei_smooth[np.isfinite(blfei_smooth) & valid_mask]
    if valid_for_otsu.size == 0:
        print(f"No valid BLFEI values for {folder_name} after masking.")
        return None

    # Use Otsu's method for thresholding
    try:
        threshold_value = threshold_otsu(valid_for_otsu)
    except Exception as e:
        print(f"Otsu thresholding failed for {folder_name}: {e}")
        return None

    blfei_classified = (blfei_smooth > threshold_value).astype(np.uint8)

    # Apply morphological operations
    structure = np.ones((3, 3))
    blfei_classified = binary_opening(blfei_classified, structure=structure)
    blfei_classified = binary_closing(blfei_classified, structure=structure).astype(np.uint8)
    blfei_classified = remove_small_objects(blfei_classified.astype(bool), min_size=100)
    blfei_classified = remove_small_holes(blfei_classified, area_threshold=100)
    blfei_classified = blfei_classified.astype(np.uint8)
    blfei_classified = median_filter(blfei_classified, size=5)

    unique_labels = np.unique(labels)
    urban_classes_present = [c for c in urban_classes.keys() if c in unique_labels]
    nonurban_classes_present = [c for c in nonurban_classes.keys() if c in unique_labels]

    # Require at least two urban classes to consider this sample
    if len(urban_classes_present) < 2:
        return None

    urban_mask = np.isin(labels, urban_classes_present).astype(np.uint8)

    # Evaluate metrics only on pixels within the valid mask
    valid_mask_flat = valid_mask.flatten()
    if not np.any(valid_mask_flat):
        print(f"No valid pixels after masking for {folder_name}.")
        return None

    blfei_flat = blfei_classified.flatten()
    urban_flat = urban_mask.flatten()

    y_pred = blfei_flat[valid_mask_flat]
    y_true = urban_flat[valid_mask_flat]

    tp_sample = np.sum((y_pred == 1) & (y_true == 1))
    fp_sample = np.sum((y_pred == 1) & (y_true == 0))
    fn_sample = np.sum((y_pred == 0) & (y_true == 1))
    tn_sample = np.sum((y_pred == 0) & (y_true == 0))

    sample_iou = tp_sample / (tp_sample + fp_sample + fn_sample) if (tp_sample + fp_sample + fn_sample) > 0 else 0
    if sample_iou < 0.6:
        return None

    # Save sample images every 50 iterations to reduce overhead
    if i % 50 == 0:
        plt.figure(figsize=(24, 8))

        # Original BLFEI
        plt.subplot(1, 3, 1)
        dbi_plot = plt.imshow(blfei, cmap="gray")
        cbar = plt.colorbar(dbi_plot, label="BLFEI Value", fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        plt.title("Original BLFEI", fontsize=14, pad=20)
        plt.axis("off")

        # Overall classification overlay
        plt.subplot(1, 3, 2)
        plt.imshow(blfei, cmap="gray")
        overlay = create_overlay(
            blfei,
            labels,
            threshold_value,
            urban_classes_present,
            nonurban_classes_present,
        )
        plt.imshow(overlay)
        urban_legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=urban_colors[c], label=urban_classes[c])
            for c in urban_classes_present
            if c in urban_colors
        ]
        nonurban_legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=nonurban_colors[c], label=nonurban_classes[c])
            for c in nonurban_classes_present
            if c in nonurban_colors
        ]
        plt.legend(
            handles=urban_legend_elements + nonurban_legend_elements,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=10,
            title="Overlay Classes",
            title_fontsize=12,
        )
        plt.title("Overall Classification Overlay", fontsize=14, pad=20)
        plt.axis("off")

        # Ground truth visualization
        plt.subplot(1, 3, 3)
        ground_truth_viz = np.zeros((*labels.shape, 3))
        for c in urban_classes_present:
            if c in urban_colors:
                ground_truth_viz[labels == c] = urban_colors[c][:3]
        for c in nonurban_classes_present:
            if c in nonurban_colors:
                ground_truth_viz[labels == c] = nonurban_colors[c][:3]
        plt.imshow(ground_truth_viz)
        ground_truth_legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=urban_colors[c], label=urban_classes[c])
            for c in urban_classes_present
            if c in urban_colors
        ]
        ground_truth_legend_elements += [
            plt.Rectangle((0, 0), 1, 1, fc=nonurban_colors[c], label=nonurban_classes[c])
            for c in nonurban_classes_present
            if c in nonurban_colors
        ]
        plt.legend(
            handles=ground_truth_legend_elements,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            fontsize=10,
            title="Ground Truth Classes",
            title_fontsize=12,
        )
        plt.title("Ground Truth", fontsize=14, pad=20)
        plt.axis("off")

        plot_path = os.path.join(output_base, f"blfei_sample_{i}.png")
        plt.tight_layout(pad=3.0)
        plt.savefig(plot_path)
        plt.close()

    return {
        "tp": tp_sample,
        "fp": fp_sample,
        "fn": fn_sample,
        "tn": tn_sample,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# -----------------------------------------------------------------------------
# Main Execution with Parallel Processing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Read the CSV only in the main process (not in workers)
    df_all = pd.read_csv(csv_path)
    paths = df_all["Path"].tolist()

    overall_tp = 0
    overall_tn = 0
    overall_fp = 0
    overall_fn = 0
    all_y_true = []
    all_y_pred = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_sample, i, path): i
            for i, path in enumerate(paths)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Error in processing sample index {idx}: {e}")
                continue

            if result is None:
                continue

            overall_tp += result["tp"]
            overall_fp += result["fp"]
            overall_fn += result["fn"]
            overall_tn += result["tn"]
            all_y_true.extend(result["y_true"])
            all_y_pred.extend(result["y_pred"])

    total_pixels_overall = overall_tp + overall_tn + overall_fp + overall_fn
    overall_accuracy = (overall_tp + overall_tn) / total_pixels_overall if total_pixels_overall else 0
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else 0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall)
        else 0
    )
    overall_iou = overall_tp / (overall_tp + overall_fp + overall_fn) if (overall_tp + overall_fp + overall_fn) else 0

    print(f"Total Pixels: {total_pixels_overall}")
    print(f"Overall True Positives: {overall_tp}")
    print(f"Overall True Negatives: {overall_tn}")
    print(f"Overall False Positives: {overall_fp}")
    print(f"Overall False Negatives: {overall_fn}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")

    if len(all_y_true) > 0 and len(all_y_pred) > 0:
        plt.figure(figsize=(8, 6))
        plot_confusion_matrix(
            all_y_true,
            all_y_pred,
            title="Overall Confusion Matrix",
            class_names=["Non-Urban", "Urban"],
        )
        confusion_matrix_path = os.path.join(output_base, "overall_confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.tight_layout()
        plt.show()

    print("Processing complete.")

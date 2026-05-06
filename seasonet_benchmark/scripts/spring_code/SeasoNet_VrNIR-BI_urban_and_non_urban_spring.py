import os
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.enums import Resampling
from rasterio import warp
from sklearn.metrics import confusion_matrix
from skimage.filters import threshold_otsu
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, median_filter
from skimage.morphology import remove_small_objects, remove_small_holes
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------  
# Define urban and non-urban classes and colors  
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
    11: "Sport and leisure facilities"
}
nonurban_classes = {
    12: "Non-irrigated arable land",
    13: "Vineyards",
    14: "Frvrnir_bit trees and berry plantations",
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
    33: "Sea and ocean"
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
    11: [0.5, 0, 0, 0.6]
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
    33: [0.0, 0.0, 0.8, 0.6]
}

# -----------------------------------------------------------------------------  
# Helper functions
# -----------------------------------------------------------------------------  
def load_and_preprocess_band(file_path, band_index):
    with rasterio.open(file_path) as src:
        band = src.read(band_index).astype(np.float32)
        band[band == src.nodata] = np.nan
        return band, src.meta

def resample_to_10m(source_path, band_index, target_meta):
    """Resample a band to 10 m resolution matching target metadata."""
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
            resampling=Resampling.bilinear
        )
    return resampled

def normalize_band(band):
    """Min-max normalization to [0, 1]."""
    band_min = np.nanmin(band)
    band_max = np.nanmax(band)
    return (band - band_min) / (band_max - band_min + 1e-6)

def find_otsu_threshold(vrnir_bi_values):
    """Find threshold using Otsu's method."""
    finite_vals = vrnir_bi_values[np.isfinite(vrnir_bi_values)]
    if len(finite_vals) < 50:
        return np.nan
    threshold = threshold_otsu(finite_vals)
    return threshold


def compute_vrnir_bi(red, nir):
    
    eps = 1e-6
    vrnir_bi = (red - nir) / (red + nir + eps)
    return vrnir_bi

def compute_mndwi(green, swir1):
    
    eps = 1e-6
    mndwi = (green - swir1) / (green + swir1 + eps)
    return mndwi

def create_overlay(vrnir_bi, labels, threshold, urban_classes_present, nonurban_classes_present):
    """Create an RGBA overlay from the thresholded vrnir_bi."""
    overlay = np.zeros((*vrnir_bi.shape, 4))
    urban_mask = vrnir_bi > threshold
    for c in urban_classes_present:
        if c in urban_colors:
            class_mask = (labels == c) & urban_mask
            overlay[class_mask] = urban_colors[c]
    for c in nonurban_classes_present:
        if c in nonurban_colors:
            class_mask = (labels == c) & (~urban_mask)
            overlay[class_mask] = nonurban_colors[c]
    return overlay

def plot_confusion_matrix(y_true, y_pred, title, class_names=None, ax=None):
    """Plot a confusion matrix with percentages."""
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    default_class_names = ['Non-Urban', 'Urban']
    if class_names is None:
        class_names = default_class_names
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    ax.set_title(title, pad=20, fontsize=14)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.2, i + 0.1, f'({cm_percent[i, j]:.1f}%)',
                    color='black', fontsize=10)
    ax.figure.tight_layout()
    return cm

# -----------------------------------------------------------------------------  
# File paths and parameters  
# -----------------------------------------------------------------------------  
csv_path = r"D:\SeasoNet\filtered_spring_meta_with_both_urban_nonurban_2.csv"
base_path = r"D:\SeasoNet\spring"
output_base = r"D:\SeasoNet\Indices Implementation\spring_results\vrnir-bi"

df_all = pd.read_csv(csv_path)
# df_all = df_all.head(100)


# -----------------------------------------------------------------------------  
# Process function  
# -----------------------------------------------------------------------------  
def process_sample(i_row):
    i, row = i_row
    folder_path_csv = row["Path"]
    folder_name = os.path.basename(folder_path_csv)
    folder_path = os.path.join(base_path, folder_path_csv)

    # Define file paths
    rgb_tif_path = os.path.join(folder_path, f"{folder_name}_10m_RGB.tif")   # contains B2, B3, B4
    nir_tif_path = os.path.join(folder_path, f"{folder_name}_10m_IR.tif")    # Band8 (NIR)
    tif_20m_path = os.path.join(folder_path, f"{folder_name}_20m.tif")
    label_tif_path = os.path.join(folder_path, f"{folder_name}_labels.tif")

    easy_mask_path   = os.path.join(folder_path, f"{folder_name}_mask_easy.tif")
    medium_mask_path = os.path.join(folder_path, f"{folder_name}_mask_medium.tif")
    clouds_path      = os.path.join(folder_path, f"{folder_name}_clouds.tif")
    snow_path        = os.path.join(folder_path, f"{folder_name}_snow.tif")

    if not all(os.path.exists(p) for p in [easy_mask_path, medium_mask_path, clouds_path, snow_path]):
        print(f"Skipping sample {folder_name}: additional mask files not found.")
        return None

    try:
        band_nir, meta_nir = load_and_preprocess_band(nir_tif_path, 1)
        band_red,_ = load_and_preprocess_band(rgb_tif_path, 3)
        band_swir_1 = resample_to_10m(tif_20m_path, 5, meta_nir)
        band_green,_ = load_and_preprocess_band(rgb_tif_path, 2)
        
        labels, _ = load_and_preprocess_band(label_tif_path, 1)
        easy_mask, _ = load_and_preprocess_band(easy_mask_path, 1)
        medium_mask, _ = load_and_preprocess_band(medium_mask_path, 1)
        clouds_mask, _ = load_and_preprocess_band(clouds_path, 1)
        snow_mask, _ = load_and_preprocess_band(snow_path, 1)

    except Exception as e:
        print(f"Error loading/resampling Sentinel-2 data for {folder_name}: {e}")
        return None


    seg_mask = ((easy_mask > 0) | (medium_mask > 0)).astype(np.uint8)
    valid_mask = (seg_mask == 1) & (clouds_mask == 0) & (snow_mask == 0)
    
    
    mndwi = compute_mndwi(band_green, band_swir_1)
    
    water_threshold = 0.3
    water_mask = mndwi > water_threshold
    valid_mask &= ~water_mask
    
    # Compute the vrnir_bi.
    vrnir_bi = compute_vrnir_bi(band_red, band_nir)
    vrnir_bi_smooth = gaussian_filter(vrnir_bi, sigma=5)
    # valid_vrnir_bi = vrnir_bi_smooth[np.isfinite(vrnir_bi_smooth)]
    # if valid_vrnir_bi.size == 0:
    #     print(f"No valid cbci values for {folder_name}.")
    #     return None
    # valid_vrnir_bi = valid_vrnir_bi.reshape(-1, 1)
    # kmeans = KMeans(n_clusters=2, random_state=42).fit(valid_vrnir_bi)
    # centers = kmeans.cluster_centers_.flatten()
    # urban_center = max(centers)
    # nonurban_center = min(centers)
    # midpoint = (urban_center + nonurban_center) / 2
    # beta = 0.0
    # biased_threshold = midpoint + beta * (urban_center - midpoint)

    valid_vrnir_bi = vrnir_bi_smooth[np.isfinite(vrnir_bi) & valid_mask]
    if valid_vrnir_bi.size < 10:
        return None
    # # Use Otsu's method for threshold selection
    T_otsu = find_otsu_threshold(valid_vrnir_bi)
    # T_otsu = 0
    vrnir_bi_classified = (vrnir_bi_smooth > T_otsu).astype(np.uint8)


# Fine-tuned morphological operations
    structure = np.ones((1, 1))  # Smaller structuring element
    vrnir_bi_classified = binary_opening(vrnir_bi_classified, structure=structure).astype(np.uint8)
    vrnir_bi_classified = binary_closing(vrnir_bi_classified, structure=structure).astype(np.uint8)
    vrnir_bi_classified = remove_small_objects(vrnir_bi_classified.astype(bool), min_size=100)  # Reduced min_size
    vrnir_bi_classified = remove_small_holes(vrnir_bi_classified, area_threshold=100)  # Reduced area_threshold
    vrnir_bi_classified = median_filter(vrnir_bi_classified, size=1)  # Smaller filter size

    unique_labels = np.unique(labels)
    urban_classes_present = [c for c in urban_classes.keys() if c in unique_labels]
    nonurban_classes_present = [c for c in nonurban_classes.keys() if c in unique_labels]
    if len(urban_classes_present) < 2:
        return None

    urban_mask = np.isin(labels, urban_classes_present).astype(np.uint8)

    tp_sample = np.sum((vrnir_bi_classified == 1) & (urban_mask == 1) & valid_mask)
    fp_sample = np.sum((vrnir_bi_classified == 1) & (urban_mask == 0) & valid_mask)
    fn_sample = np.sum((vrnir_bi_classified == 0) & (urban_mask == 1) & valid_mask)
    tn_sample = np.sum((vrnir_bi_classified == 0) & (urban_mask == 0) & valid_mask)
    denom = (tp_sample + fp_sample + fn_sample)
    sample_iou = tp_sample / denom if denom > 0 else 0
    if sample_iou < 0.6:
        return None

    if i % 90 == 0:
        vrnir_bi_min = np.nanmin(vrnir_bi)
        vrnir_bi_max = np.nanmax(vrnir_bi)
        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(vrnir_bi, cmap='gray', vmin =vrnir_bi_min, vmax = vrnir_bi_max)
        cbar = plt.colorbar(label='vrnir_bi Value', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        plt.title('vrnir_bi', fontsize=14, pad=20)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(vrnir_bi, cmap='gray', vmin = vrnir_bi_min, vmax = vrnir_bi_max)
        overlay = create_overlay(vrnir_bi, labels, T_otsu, urban_classes_present, nonurban_classes_present)
        plt.imshow(overlay)
        urban_legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=urban_colors[c], label=urban_classes[c])
            for c in urban_classes_present if c in urban_colors
        ]
        nonurban_legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=nonurban_colors[c], label=nonurban_classes[c])
            for c in nonurban_classes_present if c in nonurban_colors
        ]
        plt.legend(handles=urban_legend_elements + nonurban_legend_elements,
                   loc='upper right', bbox_to_anchor=(1.0, 1.0),
                   fontsize=10, title='Overlay Classes', title_fontsize=12)
        plt.title('vrnir_bi Classification Overlay', fontsize=14, pad=20)
        plt.axis('off')

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
        for c in urban_classes_present if c in urban_colors
        ]
        ground_truth_legend_elements += [
        plt.Rectangle((0, 0), 1, 1, fc=nonurban_colors[c], label=nonurban_classes[c])
        for c in nonurban_classes_present if c in nonurban_colors
        ]
        plt.legend(handles=ground_truth_legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0),
               fontsize=10, title='Ground Truth Classes', title_fontsize=12)
        plt.title('Ground Truth', fontsize=14, pad=20)
        plt.axis('off')

        plot_path = os.path.join(output_base, f"vrnir_bi_sample_{i}.png")
        plt.tight_layout(pad=3.0)
        plt.savefig(plot_path)
        plt.close()

    return {
        "tp": tp_sample,
        "fp": fp_sample,
        "fn": fn_sample,
        "tn": tn_sample,
        "y_true": urban_mask.flatten()[valid_mask.flatten()],
        "y_pred": vrnir_bi_classified.flatten()[valid_mask.flatten()]
    }

# Main Script
if __name__ == '__main__':
    overall_tp = 0
    overall_tn = 0
    overall_fp = 0
    overall_fn = 0
    all_y_true = []
    all_y_pred = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_sample, (i, row)): i for i, row in df_all.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            result = future.result()
            if result is None:
                continue
            overall_tp += result["tp"]
            overall_fp += result["fp"]
            overall_fn += result["fn"]
            overall_tn += result["tn"]
            all_y_true.extend(result["y_true"])
            all_y_pred.extend(result["y_pred"])

    total_pixels_overall = overall_tp + overall_tn + overall_fp + overall_fn
    if total_pixels_overall > 0:
        overall_accuracy = (overall_tp + overall_tn) / total_pixels_overall
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        overall_iou = overall_tp / (overall_tp + overall_fp + overall_fn) if (overall_tp + overall_fp + overall_fn) else 0
    else:
        overall_accuracy = overall_precision = overall_recall = overall_f1 = overall_iou = 0

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
        plot_confusion_matrix(all_y_true, all_y_pred, title="vrnir_bi Overall Confusion Matrix", class_names=['Non-Urban', 'Urban'])
        confusion_matrix_path = os.path.join(output_base, "overall_confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.tight_layout()
        plt.show()

    print("vrnir_bi Processing complete.")
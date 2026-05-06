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
from scipy.ndimage import binary_opening, binary_closing, median_filter
from skimage.morphology import remove_small_objects, remove_small_holes
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
    11: "Sport and leisure facilities"
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
# Helper Functions for Data Processing and Visualization
# -----------------------------------------------------------------------------
def load_and_preprocess_band(file_path, band_index):
    """Load and preprocess a single band from a raster file."""
    with rasterio.open(file_path) as src:
        band = src.read(band_index).astype(np.float32)
        band[band == src.nodata] = np.nan  # Replace nodata values with NaN
        return band, src.meta

def resample_to_10m(source_path, band_index, target_meta):
    """Resample a band to 10m resolution matching target metadata."""
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

def calculate_ndbi_blue_green(blue, green):
    """Calculate NDBI using Blue and Green bands."""
    epsilon = 1  # Small constant to avoid division by zero
    return (blue - green) / (blue + green + epsilon)

def calculate_ndbi_red_green(red, green):
    """Calculate NDBI using Red and Green bands."""
    epsilon = 1
    return (red - green) / (red + green + epsilon)

def calculate_bbi_binary(ndbi_blue_green, ndbi_red_green):
    """Calculate BBI as the sum of binary indices, per the paper's method."""
    ndbi_bg_binary = (ndbi_blue_green > 0).astype(np.uint8)  # Binary: 1 if > 0, else 0
    ndbi_rg_binary = (ndbi_red_green > 0).astype(np.uint8)  # Binary: 1 if > 0, else 0
    bbi = ndbi_bg_binary + ndbi_rg_binary  # Sum of binary values (0, 1, or 2)
    return bbi


def create_overlay(bbi, labels, urban_classes_present, nonurban_classes_present):
    """Create an overlay image for visualization using BBI classification."""
    overlay = np.zeros((*bbi.shape, 4))  # RGBA array
    urban_mask = bbi > 0  # Classify as urban where BBI > 0
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

def compute_excess_green(r, g, b):
    """
    A simple vegetation indicator when only R/G/B are available.
    Typical vegetation will have ExG > 0. 
    Adjust threshold or formula as needed.
    """
    return 2.0 * g - r - b

def compute_blue_ratio(blue, red, green, epsilon=1e-6):
    """
    Compute a simple ratio of Blue / (Red + Green) to approximate water detection.
    We add a small epsilon to avoid dividing by zero.
    """
    return blue / (red + green + epsilon)

# -----------------------------------------------------------------------------
# Set Paths
# -----------------------------------------------------------------------------
csv_path = r"D:\SeasoNet\filtered_summer_meta_with_both_urban_nonurban_2.csv"
base_path = r"F:\SeasoNet"
output_base = r"D:\SeasoNet\Indices Implementation\summer_results\bbi"


# Read the entire CSV file and limit to first 100 rows
df_all = pd.read_csv(csv_path)
# df_all = df_all.head(10000)

# -----------------------------------------------------------------------------
# Initialize overall metrics counters
# -----------------------------------------------------------------------------
overall_tp = 0
overall_tn = 0
overall_fp = 0
overall_fn = 0
all_y_true = []
all_y_pred = []

# -----------------------------------------------------------------------------
# Process every row in the CSV file for BBI evaluation with a progress bar
# -----------------------------------------------------------------------------
def process_sample(i_row):
    i, row = i_row
    row = df_all.iloc[i]
    folder_path_csv = row['Path']
    folder_name = os.path.basename(folder_path_csv)
    folder_path = os.path.join(base_path, folder_path_csv)

    rgb_tif_path = os.path.join(folder_path, f"{folder_name}_10m_RGB.tif")
    label_tif_path = os.path.join(folder_path, f"{folder_name}_labels.tif")

    # Load additional mask files
    easy_mask_path = os.path.join(folder_path, f"{folder_name}_mask_easy.tif")
    medium_mask_path = os.path.join(folder_path, f"{folder_name}_mask_medium.tif")
    clouds_path = os.path.join(folder_path, f"{folder_name}_clouds.tif")
    snow_path = os.path.join(folder_path, f"{folder_name}_snow.tif")
    if not (os.path.exists(easy_mask_path) and os.path.exists(medium_mask_path) and 
            os.path.exists(clouds_path) and os.path.exists(snow_path)):
        print(f"Skipping sample {folder_name}: additional mask files not found.")
        return None

    try:
        band_red, _ = load_and_preprocess_band(rgb_tif_path, 3)  # Red band
        band_green, _ = load_and_preprocess_band(rgb_tif_path, 2)  # Green band
        band_blue, _ = load_and_preprocess_band(rgb_tif_path, 1)  # Blue band
        labels, _ = load_and_preprocess_band(label_tif_path, 1)
        # Load the additional masks
        easy_mask, _ = load_and_preprocess_band(easy_mask_path, 1)
        medium_mask, _ = load_and_preprocess_band(medium_mask_path, 1)
        clouds_mask, _ = load_and_preprocess_band(clouds_path, 1)
        snow_mask, _ = load_and_preprocess_band(snow_path, 1)
    except Exception as e:
        print(f"Error loading data for {folder_name}: {e}")
        return None

    # Create a combined segmentation mask and a valid mask
    seg_mask = ((easy_mask > 0) | (medium_mask > 0)).astype(np.uint8)
    valid_mask = (seg_mask == 1) & (clouds_mask == 0) & (snow_mask == 0)

    # Calculate NDBI_Blue-Green and NDBI_Red-Green
    ndbi_blue_green = calculate_ndbi_blue_green(band_blue, band_green)
    ndbi_red_green = calculate_ndbi_red_green(band_red, band_green)

    # Calculate BBI using the binary sum method from the paper (for classification)
    bbi = calculate_bbi_binary(ndbi_blue_green, ndbi_red_green)

    # Classify based on BBI > 0 (paper's method)
    bbi_classified_1 = (bbi > 0).astype(np.uint8)
    
    exg = compute_excess_green(band_red, band_green, band_blue)
    vegetation_mask = (exg > 0)  # True where likely vegetation
    bbi_classified_1[vegetation_mask] = 0
    
    #NEW STEP: Water mask using Blue Ratio
    #   If blue is significantly higher than red+green, label as water (non-urban)
    blue_ratio = compute_blue_ratio(band_blue, band_red, band_green)
    
    # You can try different thresholds or combine with absolute reflectance checks.
    # Example: require blue > 0.1 to avoid super-dark pixels, and ratio > 1.0
    water_mask = (blue_ratio > 1.0) & (band_blue > 0.1)
    
    bbi_classified =bbi_classified_1
    
    # Force these pixels to be non-urban
    bbi_classified[water_mask] = 0

    # Apply morphological operations with a larger kernel
    structure = np.ones((1,1))
    bbi_classified = binary_opening(bbi_classified, structure=structure).astype(np.uint8)
    bbi_classified = binary_closing(bbi_classified, structure=structure).astype(np.uint8)
    bbi_classified = remove_small_objects(bbi_classified.astype(bool), min_size=500)
    bbi_classified = remove_small_holes(bbi_classified, area_threshold=500)
    bbi_classified = bbi_classified.astype(np.uint8)
    bbi_classified = median_filter(bbi_classified, size=1)

    unique_labels = np.unique(labels)
    urban_classes_present = [c for c in urban_classes.keys() if c in unique_labels]
    nonurban_classes_present = [c for c in nonurban_classes.keys() if c in unique_labels]

    if len(urban_classes_present) < 2:
        return None

    urban_mask = np.isin(labels, urban_classes_present).astype(np.uint8)

    height, width = bbi.shape
    sample_total_pixels = height * width

    # Evaluate metrics only on pixels within the valid mask
    tp_sample = np.sum((bbi_classified == 1) & (urban_mask == 1) & valid_mask)
    fp_sample = np.sum((bbi_classified == 1) & (urban_mask == 0) & valid_mask)
    fn_sample = np.sum((bbi_classified == 0) & (urban_mask == 1) & valid_mask)
    tn_sample = np.sum((bbi_classified == 0) & (urban_mask == 0) & valid_mask)
    sample_iou = tp_sample / (tp_sample + fp_sample + fn_sample) if (tp_sample + fp_sample + fn_sample) > 0 else 0

    if sample_iou < 0.6:
        return None

    # Save sample images for every iteration (adjustable)
    if i % 90 == 0:
        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        bbi_plot = plt.imshow(bbi, cmap='gray', vmin=0, vmax=2)  # BBI range: 0 to 2
        cbar = plt.colorbar(bbi_plot, label='BBI Value', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        plt.title('BBI', fontsize=14, pad=20)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(bbi, cmap='gray', vmin=0, vmax=2)
        overlay = create_overlay(bbi, labels, urban_classes_present, nonurban_classes_present)
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
        plt.title('Overall Classification Overlay', fontsize=14, pad=20)
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


        plot_path = os.path.join(output_base, f"bbi_sample_{i}.png")
        plt.tight_layout(pad=3.0)
        plt.savefig(plot_path)
        plt.close()

    return {
        "tp": tp_sample,
        "fp": fp_sample,
        "fn": fn_sample,
        "tn": tn_sample,
        "y_true": urban_mask.flatten()[valid_mask.flatten()],
        "y_pred": bbi_classified.flatten()[valid_mask.flatten()]
    }

# End for loop

if __name__ == '__main__':
    overall_tp = 0
    overall_tn = 0
    overall_fp = 0
    overall_fn = 0
    all_y_true = []
    all_y_pred = []

    # Use ProcessPoolExecutor to parallelize processing.
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
    overall_accuracy = (overall_tp + overall_tn) / total_pixels_overall if total_pixels_overall else 0
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) else 0
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
        plot_confusion_matrix(all_y_true, all_y_pred, title="Overall Confusion Matrix", class_names=['Non-Urban', 'Urban'])
        confusion_matrix_path = os.path.join(output_base, "overall_confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.tight_layout()
        plt.show()

    print("Processing complete.")
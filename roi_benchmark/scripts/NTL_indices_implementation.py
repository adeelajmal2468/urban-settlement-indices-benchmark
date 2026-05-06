import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    jaccard_score, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, cohen_kappa_score,
    roc_curve, auc
)
from skimage.transform import resize
import pandas as pd
import matplotlib.patches as mpatches
import os

# ======= Settings =======
landsat_path = r'C:\Users\Adeel Ajmal\Downloads\L8_BahriaPhase7_20231012.tif'
label_path   = r'C:\Users\Adeel Ajmal\Downloads\DynamicWorld_LandcoverLabels.tif'
viirs_path   = r'C:\Users\Adeel Ajmal\Downloads\VIIRS_NTL_90p_BahriaPhase7_20230101_20250701.tif' # Update if needed
evi_max_path = r'C:\Users\Adeel Ajmal\Downloads\L8_EVImax_BahriaPhase7_20230101_20250701.tif'
ndvi_max_path = r'C:\Users\Adeel Ajmal\Downloads\L8_NDVImax_BahriaPhase7_20230101_20250701.tif'
viirs_rgb_path = r'C:\Users\Adeel Ajmal\Downloads\VIIRS_NTL_RGB_BahriaPhase7_2023.tif'
ubi_max_path = r'C:\Users\Adeel Ajmal\Downloads\UBImax_BahriaPhase7_20230101_20250701.tif'
outdir       = "urban_index_eval_NTL_all"
save_figs    = True
os.makedirs(outdir, exist_ok=True)

# ======= 1. Load Data =======
with rasterio.open(landsat_path) as src:
    stack = src.read().astype('float32')
b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11 = stack

with rasterio.open(label_path) as src:
    labels = src.read(1)

with rasterio.open(viirs_path) as src:
    viirs_ntl = src.read(1).astype('float32')
    

# Load annual max composites
with rasterio.open(evi_max_path) as src:
    EVI_max = src.read(1).astype('float32')
with rasterio.open(ndvi_max_path) as src:
    NDVI_max = src.read(1).astype('float32')
    
with rasterio.open(viirs_rgb_path) as src:
    ntl_jan, ntl_may, ntl_sep = src.read()
# Normalize each band to [0, 1]
def norm(x): return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-6)
ntl_jan_norm = norm(ntl_jan)
ntl_may_norm = norm(ntl_may)
ntl_sep_norm = norm(ntl_sep)
ntl_rgb = np.stack([ntl_jan_norm, ntl_may_norm, ntl_sep_norm], axis=-1)


with rasterio.open(ubi_max_path) as src:
    UBI_max = src.read(1).astype('float32')

gt_mask = (labels == 6).astype(np.uint8)
gt_mask_resized = resize(
    gt_mask,
    (stack.shape[1], stack.shape[2]),
    order=0, preserve_range=True, anti_aliasing=False
).astype(np.uint8)

# ======= 2. Define Indices (as per docx) =======
# Note: Add all requested indices here. All formulas are implemented below.
# For missing bands, formulas are adapted for Landsat 8. Each index is nan-masked if denominator can be zero.
def safe_div(num, denom):
    out = np.full_like(num, np.nan)
    valid = np.abs(denom) > 1e-6
    out[valid] = num[valid] / denom[valid]
    return out

indices = {}


# ---------- EXTRA PREP: Reflectance ratio bands (already float32) ----------
# Assumption: stack contains TOA reflectance for optical, TOA BT (Kelvin) for thermal.
rho_red  = b4          # reflectance red
rho_nir  = b5
rho_swir1= b6
rho_green= b3
rho_swir2= b7
rho_blue = b2




# ---------- 1. NDVI, MNDWI (unchanged) ----------
NDVI = safe_div(rho_nir - rho_red, rho_nir + rho_red)
MNDWI = safe_div(rho_green - rho_swir1, rho_green + rho_swir1)
NDWI = safe_div(rho_green - rho_nir, rho_green + rho_nir)

indices['NDVI'] = NDVI
indices['MNDWI'] = MNDWI

# ======= 4. Normalize VIIRS NTL =======
ntl_mask = np.isfinite(viirs_ntl) & (viirs_ntl > 0)
NTL_min = np.nanmin(viirs_ntl[ntl_mask])
NTL_max = np.nanmax(viirs_ntl[ntl_mask])
NTL_norm = np.zeros_like(viirs_ntl)
NTL_norm[ntl_mask] = (viirs_ntl[ntl_mask] - NTL_min) / (NTL_max - NTL_min)
NTL_norm[~ntl_mask] = 0

# Masked NTL for urban (optional, not used in index formulas, but you can analyze separately)
NTL_urban = NTL_norm * gt_mask_resized


# ======= Implement Urban NTL Indices (add to indices dictionary) =======

# 1. HSI: Human Settlement Index
# HSI = ((1 - NDVI_max) + NTL_norm) / ((1 - NTL_norm) + NDVI_max + NTL_norm * NDVI_max)
indices["HSI"] = safe_div((1 - NDVI_max) + NTL_norm, (1 - NTL_norm) + NDVI_max + NTL_norm * NDVI_max)

# # 2. VANUI: Vegetation Adjusted NTL Urban Index
indices["VANUI"] = (1 - NDVI) * NTL_norm

# # 3. LISI: Large-scale Impervious Surface Index
indices["LISI"] = (1 - NDVI_max) * np.sqrt(NTL_norm)

# # 4. NDUI: Normalized Difference Urban Index
# # NDUI = (NTL_norm - NDVI) / (NTL_norm + NDVI), NDVI >= 0
ndui_mask = (NDVI >= 0)
indices["NDUI"] = np.full_like(NDVI, np.nan)
indices["NDUI"][ndui_mask] = safe_div(NTL_norm[ndui_mask] - NDVI[ndui_mask], NTL_norm[ndui_mask] + NDVI[ndui_mask])

# 5. UBI: Urban Built-Up Index (Hue/Value method)
from matplotlib.colors import rgb_to_hsv

# c * Red: use c=2 as common in literature
c = 2.0
syn_rgb = np.stack([rho_nir, rho_swir1, c * rho_red], axis=-1)
# Normalize RGB to [0,1] for HSV
rgb_min = np.nanmin(syn_rgb, axis=(0,1))
rgb_max = np.nanmax(syn_rgb, axis=(0,1))
rgb_norm = (syn_rgb - rgb_min) / (rgb_max - rgb_min + 1e-6)
hsv = rgb_to_hsv(rgb_norm)
H = hsv[...,0]
V = hsv[...,2]
indices["UBI"] = safe_div(H - V, H + V)

# 6. EUBI: Enhanced Urban Built-up Index (EUBI = NTL_max / UBI_max)
EUBI = safe_div(NTL_norm, UBI_max)
indices["EUBI"] = EUBI

# # 7. NUACI: Normalized Urban Areas Composite Index

NDWI = safe_div(rho_green - rho_nir, rho_green + rho_nir)


# The Enhanced Vegetation Index (EVI) formula is: EVI = G * ((NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)). This formula uses near-infrared (NIR), red (Red), and blue (Blue) reflectance values from remote sensing data. Constants G, C1, C2, and L are scaling factors used to adjust for atmospheric and background noise. 
# Here's a breakdown of the components: 
# NIR: Reflectance in the near-infrared band.
# Red: Reflectance in the red band.
# Blue: Reflectance in the blue band.
# G: Gain factor, usually 2.5.
# C1: Aerosol resistance term, usually 6.
# C2: Aerosol resistance term, usually 7.5.
# L: Canopy background adjustment, usually 1.
# The EVI is designed to be more sensitive to changes in vegetation and less susceptible to atmospheric effects than other vegetation indices like NDVI. It is particularly useful in areas with dense vegetation and for reducing the impact of atmospheric conditions. 

# --- After band assignment ---
G = 2.5
C1 = 6.0
C2 = 7.5
L = 1.0
EVI = G * (rho_nir - rho_red) / (rho_nir + C1 * rho_red - C2 * rho_blue + L)
indices['EVI'] = EVI

ndwi_urban = NDWI[gt_mask_resized == 1]
evi_max_urban = EVI_max[gt_mask_resized == 1]  # <--- use annual max EVI here!

a = np.nanmean(ndwi_urban)
b = np.nanmean(evi_max_urban)
r_base = np.sqrt(np.nanvar(ndwi_urban) + np.nanvar(evi_max_urban))

print(f"NUACI centroid: a={a:.4f}, b={b:.4f}, radius={r_base:.4f}")
# r_factors = [1.0, 1.25, 1.5, 2.0, 2.25, 2.5, 3, 3.2, 3.5, 4, 4.2, 4.5, 5]  # Try original and larger radii
radius_factor = 2.25  # <<< Use this value, based on your IoU/F1 analysis
r = r_base * radius_factor

print(f"NUACI centroid: a={a:.4f}, b={b:.4f}, radius={r:.4f} (factor {radius_factor})")

# --- Compute NUACI with this radius ---
d = np.sqrt((NDWI - a)**2 + (EVI_max - b)**2)
cond = (d <= r)
OLS = NTL_norm
OLS_min = np.nanmin(OLS)
OLS_max = np.nanmax(OLS)
norm_OLS = (OLS - OLS_min) / (OLS_max - OLS_min + 1e-6)

NUACI = np.zeros_like(NDWI)
NUACI[cond] = (1 - d[cond] / r) * norm_OLS[cond]

indices["NUACI"] = NUACI

# from collections import OrderedDict
# nuaci_stats = OrderedDict()

# for factor in r_factors:
#     r = r_base * factor
#     d = np.sqrt((NDWI - a)**2 + (EVI_max - b)**2)
#     cond = (d <= r)
#     # Normalize OLS for the full image
#     OLS = NTL_norm
#     OLS_min = np.nanmin(OLS)
#     OLS_max = np.nanmax(OLS)
#     norm_OLS = (OLS - OLS_min) / (OLS_max - OLS_min + 1e-6)
#     NUACI = np.zeros_like(NDWI)
#     NUACI[cond] = (1 - d[cond] / r) * norm_OLS[cond]

#     thr = best_threshold(NUACI, gt_mask_resized)
#     pred = ((NUACI > thr) & (NDVI < 0.50)).astype(np.uint8)
#     stats = metrics(pred, gt_mask_resized, idx_pred_scores=NUACI)
#     nuaci_stats[factor] = stats
#     print(f"Radius = {r:.3f} (factor {factor}): IoU={stats['IoU']:.3f}, Precision={stats['Pr']:.3f}, Recall={stats['Re']:.3f}, F1={stats['F1']:.3f}, TP={stats['TP']}")
# # --- Optional: Print all results together ---
# print("\nRadius Factor |    IoU   | Precision | Recall   |    F1   |    TP")
# for factor, stats in nuaci_stats.items():
#     print(f"{factor:12} | {stats['IoU']:.4f} |  {stats['Pr']:.4f} | {stats['Re']:.4f} | {stats['F1']:.4f} | {stats['TP']}")




# # 8. NAISI: Nighttime Lights Adjusted Impervious Surface Index
# # For simplicity, you can skip PC1 and use NTL_norm
# # Tasseled Cap TC3 (need to calculate with Landsat coefficients)
from sklearn.decomposition import PCA
h, w, c = ntl_rgb.shape
flat = ntl_rgb.reshape(-1, 3)
flat_no_nan = np.nan_to_num(flat, nan=0.0)  # or use mean imputation if desired
pc1 = PCA(n_components=1).fit_transform(flat_no_nan).reshape(h, w)
pc1_mask = np.all(np.isfinite(ntl_rgb), axis=-1)
pc1[~pc1_mask] = np.nan
PC1_norm = (pc1 - np.nanmin(pc1)) / (np.nanmax(pc1) - np.nanmin(pc1) + 1e-6)


TC3 = 0.1511*rho_blue + 0.1973*rho_green + 0.3283*rho_red + 0.3407*rho_nir - 0.7117*rho_swir1 - 0.4559*rho_swir2
SAVI = safe_div(1.5 * (rho_nir - rho_red), rho_nir + rho_red + 0.5)
SAVI_norm = (SAVI - np.nanmin(SAVI)) / (np.nanmax(SAVI) - np.nanmin(SAVI) + 1e-6)
TC3_norm = (TC3 - np.nanmin(TC3)) / (np.nanmax(TC3) - np.nanmin(TC3) + 1e-6)
NAISI = PC1_norm + TC3_norm - 2*SAVI_norm / 2

indices["NAISI"] = NAISI


# ======= 7. NDVI Threshold Search & Best Mask =======
ndvi_thresholds = np.arange(0.1, 0.81, 0.05)
best_results = {}
all_results = {}

def best_threshold(idx, truth, n_steps=200):
    v = np.isfinite(idx) & np.isfinite(truth)
    x, y = idx[v].ravel(), truth[v].ravel()
    best_thr, best_iou = None, -1
    for thr in np.linspace(np.nanmin(x), np.nanmax(x), n_steps):
        y_pred = (x > thr).astype(np.uint8)
        if y_pred.sum() == 0: continue
        iou = jaccard_score(y, y_pred, zero_division=0)
        if iou > best_iou: best_thr, best_iou = thr, iou
    return best_thr

def metrics(mask_pred, truth, idx_pred_scores=None):
    v = np.isfinite(mask_pred) & np.isfinite(truth)
    y_pred, y_true = mask_pred[v].ravel(), truth[v].ravel()
    out = dict(
        IoU=jaccard_score(y_true, y_pred, zero_division=0),
        F1=f1_score(y_true, y_pred, zero_division=0),
        Pr=precision_score(y_true, y_pred, zero_division=0),
        Re=recall_score(y_true, y_pred, zero_division=0),
        Acc=accuracy_score(y_true, y_pred),
        BAc=balanced_accuracy_score(y_true, y_pred),
        Kappa=cohen_kappa_score(y_true, y_pred)
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    out.update({"TP":tp,"TN":tn,"FP":fp,"FN":fn})
    if idx_pred_scores is not None:
        v2 = v & np.isfinite(idx_pred_scores)
        y_true2 = truth[v2].ravel()
        y_score2 = idx_pred_scores[v2].ravel()
        if len(np.unique(y_true2)) > 1 and len(y_score2) > 0:
            try:
                fpr, tpr, _ = roc_curve(y_true2, y_score2)
                out["AUC"] = auc(fpr, tpr)
            except Exception as e:
                out["AUC"] = np.nan
        else:
            out["AUC"] = np.nan
    else:
        out["AUC"] = np.nan
    return out

for idx_name, idx in indices.items():
    metric_by_thr = []
    for ndvi_thr in ndvi_thresholds:
        thr = best_threshold(idx, gt_mask_resized)
        pred = ((idx > thr) & (NDVI < ndvi_thr)).astype(np.uint8)
        result = metrics(pred, gt_mask_resized, idx_pred_scores=idx)
        result['ndvi_thr'] = ndvi_thr
        result['idx_thr'] = thr
        metric_by_thr.append(result)
    df_thr = pd.DataFrame(metric_by_thr)
    best_i = df_thr['IoU'].idxmax()
    best_results[idx_name] = df_thr.loc[best_i].to_dict()
    all_results[idx_name] = df_thr

# ======= 8. Visualization: 3-panel for Each Index =======
class_palette = [
    "#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635",
    "#dfc35a", "#c4281b", "#a59b8f", "#ffffff"
]
class_names = [
    "Water", "Trees", "Grass", "Flooded Veg", "Crops",
    "Shrub/Scrub", "Built", "Bare", "Snow/Ice"
]
label_viz = resize(labels, (stack.shape[1], stack.shape[2]), order=0, preserve_range=True, anti_aliasing=False)

rgb = np.stack([b4, b3, b2], -1)
p2, p98 = np.nanpercentile(rgb, (2, 98))
rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
rgb[np.isnan(rgb)] = 0

URBAN_COLOR = np.array([1.0, 0.1, 0.1])
NON_URBAN_COLOR = np.array([0.65, 0.65, 0.65])

for idx_name, idx in indices.items():
    best = best_results[idx_name]
    ndvi_thr = best['ndvi_thr']
    idx_thr = best['idx_thr']
    pred = ((idx > idx_thr) & (NDVI < ndvi_thr)).astype(np.uint8)
    
    fig, axs = plt.subplots(1, 3, figsize=(22, 7))
    axs[0].imshow(rgb)
    axs[0].set_title("Landsat-8 RGB")
    axs[0].axis('off')
    
    im2 = axs[1].imshow(label_viz, cmap=ListedColormap(class_palette), vmin=0, vmax=8)
    axs[1].set_title("Dynamic World Labels")
    axs[1].axis('off')
    cbar = plt.colorbar(im2, ax=axs[1], ticks=np.arange(0,9), shrink=0.8)
    cbar.ax.set_yticklabels(class_names)
    
    idx_disp = np.copy(idx)
    idx_disp[~np.isfinite(idx_disp)] = np.nan
    axs[2].imshow(idx_disp, cmap='RdYlBu')
    gt_viz = np.zeros((*gt_mask_resized.shape, 3), dtype=np.float32)
    gt_viz[gt_mask_resized == 1] = URBAN_COLOR
    gt_viz[gt_mask_resized == 0] = NON_URBAN_COLOR
    pred_viz = np.zeros((*pred.shape, 3), dtype=np.float32)
    pred_viz[pred == 1] = URBAN_COLOR
    pred_viz[pred == 0] = NON_URBAN_COLOR
    axs[2].imshow(0.6*gt_viz + 0.4*pred_viz, alpha=0.45)
    axs[2].set_title(f"{idx_name} Index\nIoU={best['IoU']:.3f} NDVI<{ndvi_thr:.2f}")
    axs[2].axis('off')
    handles = [
        mpatches.Patch(color=URBAN_COLOR, label='Urban'),
        mpatches.Patch(color=NON_URBAN_COLOR, label='Non-Urban')
    ]
    axs[2].legend(handles=handles, loc='lower right', fontsize=12)
    plt.tight_layout()
    if save_figs:
        plt.savefig(os.path.join(outdir, f"{idx_name}_urban3panel.png"), dpi=200, bbox_inches='tight')
    plt.show()

# ======= 9. Save Summary Table =======
header = "{:<10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6} {:>8}".format(
         "Index","IoU","F1","Pr","Re","Acc","BAc","Kappa","AUC","TP","NDVI_thr")
print(header)
print("-"*len(header))
for k,v in best_results.items():
    print("{:<10} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:6.0f} {:8.2f}".format(
        k, v['IoU'], v['F1'], v['Pr'], v['Re'], v['Acc'], v['BAc'], v['Kappa'], v['AUC'], v['TP'], v['ndvi_thr']))
df = pd.DataFrame(best_results).T
df.to_csv(os.path.join(outdir, "urban_index_metrics_best.csv"))
for k, df_thr in all_results.items():
    df_thr.to_csv(os.path.join(outdir, f"{k}_ndvi_threshold_sweep.csv"))
print("GT Urban Fraction: {:.2%}".format(gt_mask_resized.mean()))
vals, counts = np.unique(label_viz.astype(int), return_counts=True)
total = label_viz.size
print("\n--- Land Cover Breakdown (Dynamic World) ---")
for v, c in zip(vals, counts):
    print(f"{class_names[v]:<12}: {c/total*100:5.2f}%")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def plot_confusion_matrix(y_true, y_pred, idx_name, outdir, labels=['Non-Urban', 'Urban']):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{idx_name} - Confusion Matrix')
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(outdir, f"{idx_name}_confusion_matrix.png"), dpi=150)
    plt.close()

# Main loop: Go through all indices
for idx_name, idx in indices.items():
    best = best_results[idx_name]
    ndvi_thr = best['ndvi_thr']
    idx_thr = best['idx_thr']
    pred = ((idx > idx_thr) & (NDVI < ndvi_thr)).astype(np.uint8)
    y_true = gt_mask_resized.ravel()
    y_pred = pred.ravel()
    plot_confusion_matrix(y_true, y_pred, idx_name, outdir)

print(f"Confusion matrix PNGs saved to {outdir}")

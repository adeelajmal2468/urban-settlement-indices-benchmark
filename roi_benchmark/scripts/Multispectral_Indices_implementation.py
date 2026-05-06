import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, rgb_to_hsv
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
landsat_path = r'D:\Urban\Bahria Dataset\L8_BahriaPhase7_20231012.tif'
label_path   = r'D:\Urban\Bahria Dataset\DynamicWorld_LandcoverLabels.tif'
outdir       = "urban_index_eval_multispectral_all"
save_figs    = True
os.makedirs(outdir, exist_ok=True)

# ======= 1. Load Data =======
with rasterio.open(landsat_path) as src:
    stack = src.read().astype('float32')
b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11 = stack

with rasterio.open(label_path) as src:
    labels = src.read(1)

gt_mask = (labels == 6).astype(np.uint8)
gt_mask_resized = resize(
    gt_mask,
    (stack.shape[1], stack.shape[2]),
    order=0, preserve_range=True, anti_aliasing=False
).astype(np.uint8)

# ======= 2. Define Indices (25 multispectral from paper) =======
def safe_div(num, denom, eps=1e-6):
    out = np.full_like(num, np.nan, dtype=np.float32)
    valid = np.isfinite(num) & np.isfinite(denom) & (np.abs(denom) > eps)
    out[valid] = num[valid] / denom[valid]
    return out

def minmax_norm(x, eps=1e-6):
    x = x.astype(np.float32)
    v = np.isfinite(x)
    if not np.any(v):
        return np.zeros_like(x, dtype=np.float32)
    xmin = np.nanmin(x[v])
    xmax = np.nanmax(x[v])
    return (x - xmin) / ( (xmax - xmin) + eps )

# Landsat-8 reflectance shorthand
rho_blue  = b2
rho_green = b3
rho_red   = b4
rho_nir   = b5
rho_swir1 = b6
rho_swir2 = b7


def compute_tasseled_cap(stack, coeff_brightness, coeff_greenness, coeff_wetness):
    """
    Compute Tasseled Cap components using a dot product along the bands.
    stack: numpy array of shape (bands, H, W)
    coeff_*: 1D numpy array of length equal to number of bands.
    Returns TC_brightness, TC_greenness, TC_wetness (each with shape (H, W))
    """
    H, W = stack.shape[1], stack.shape[2]
    stack_2d = stack.reshape(stack.shape[0], -1)  # [bands, H*W]
    TC_brightness = np.dot(coeff_brightness, stack_2d).reshape(H, W)
    TC_greenness  = np.dot(coeff_greenness, stack_2d).reshape(H, W)
    TC_wetness    = np.dot(coeff_wetness, stack_2d).reshape(H, W)
    return TC_brightness, TC_greenness, TC_wetness

# Core helpers used by several indices
NDVI  = safe_div(rho_nir - rho_red , rho_nir + rho_red)          # (eq. 32 context)
ndvi  = NDVI  # keep alias used later in your evaluation code
NDWI  = safe_div(rho_green - rho_nir, rho_green + rho_nir)       # (eq. 41)
MNDWI = safe_div(rho_green - rho_swir1, rho_green + rho_swir1)   # (eq. 45)
SAVI  = safe_div((1.0 + 0.5) * (rho_nir - rho_red) , (rho_nir + rho_red + 0.5))  # (eq. 31; L=0.5)

# Landsat-8 Tasseled Cap (Baig et al. coefficients, used in paper)
TC1 = ((0.3443*rho_blue) + (0.4057*rho_green) + (0.4667*rho_red) +
       (0.5347*rho_nir) + (0.3936*rho_swir1) + (0.2412*rho_swir2))
TC2 = ((-0.2365*rho_blue) +(- 0.2836*rho_green) + (- 0.4257*rho_red) +
        (0.8097*rho_nir) + (0.0043*rho_swir1) + (- 0.1638*rho_swir2))
TC3 = ((0.1301*rho_blue) + (0.2280*rho_green) + (0.3492*rho_red) +
       (0.1795*rho_nir) + (- 0.6270*rho_swir1) +  (- 0.6195*rho_swir2))  # (eq. 74 for wetness)

# Normalized H/V/L (brightness/greenness/wetness) for BCI, RNDSI
H = minmax_norm(TC1)
V = minmax_norm(TC2)
L = minmax_norm(TC3)

# PCA first component for CBI (over R,G,NIR)
from sklearn.decomposition import PCA
def first_pc_3(a, b, c):
    h, w = a.shape
    X = np.stack([a.ravel(), b.ravel(), c.ravel()], axis=1)
    pc1 = PCA(n_components=1).fit_transform(np.nan_to_num(X)).reshape(h, w).astype(np.float32)
    return pc1

PC1_rgN = first_pc_3(rho_red, rho_green, rho_nir)
PC1n = minmax_norm(PC1_rgN)

# Build indices dict
indices = {}

# 1) NDBI (eq. 1)
indices["NDBI"] = safe_div(rho_swir1 - rho_nir, rho_swir1 + rho_nir)

# 2a) IBI (NDVI version) (eq. 29)
indices["IBI_NDVI"] = safe_div(2.0*(indices["NDBI"] - (NDVI + MNDWI)),
                               2.0*(indices["NDBI"] + (NDVI + MNDWI)))

# 2b) IBI (SAVI version) (eq. 30,31)
indices["IBI_SAVI"] = safe_div(2.0*(indices["NDBI"] - (SAVI + MNDWI)),
                               2.0*(indices["NDBI"] + (SAVI + MNDWI)))

# 3) BAEI (eq. 18)
indices["BAEI"] = safe_div(rho_red + 0.3, rho_green + rho_swir1)

# 4) BBI (eq. 7) – continuous form so your threshold search still works
NDBI_BG = safe_div(rho_blue - rho_green , rho_blue + rho_green)
NDBI_RG = safe_div(rho_red  - rho_green , rho_red  + rho_green)
# NDBI_BG_binary = (NDBI_BG > 0).astype(np.uint8)  # Binary: 1 if > 0, else 0
# NDBI_RG_binary = (NDBI_RG > 0).astype(np.uint8)  # Binary: 1 if > 0, else 0
indices["BBI"] = NDBI_BG + NDBI_RG

# 5) BCI (eq. 33) with H,V,L from TCT (min–max normalised)
# BCI = ((H + L)/2 - V) / ((H + L)/2 + V)
HL_mean = 0.5*(H + L)
indices["BCI"] = safe_div(HL_mean - V, HL_mean + V)

# 6) BLFEI (eq. 43)
indices["BLFEI"] = safe_div((rho_green + rho_red + rho_swir2)/3.0 - rho_swir1,
                            (rho_green + rho_red + rho_swir2)/3.0 + rho_swir1)

# 7) BUb (eq. 27) — binary-difference style (values in {-254,0,254})
NDBI_bin = (indices["NDBI"] > 0).astype(np.float32)
NDVI_bin = (NDVI > 0).astype(np.float32)
indices["BUb"] = 254.0*(NDBI_bin - NDVI_bin)

# 8) BUc (eq. 28)
indices["BUc"] = safe_div(rho_swir1 - rho_nir, rho_swir1 + rho_nir) - safe_div(rho_nir - rho_red, rho_nir + rho_red)

# 9) BUI (eq. 19)
indices["BUI"] = safe_div(rho_red - rho_swir1 , rho_red + rho_swir1) + \
                 safe_div(rho_swir2 - rho_swir1, rho_swir2 + rho_swir1)

# 10) CBCI (eq. 9) with MBSI (eq. 9) and OSAVI (eq. 10)
MBSI  = safe_div(2*(rho_red - rho_green), (rho_red + rho_green)-2)
OSAVI = safe_div(rho_nir - rho_red, rho_nir + rho_red + 0.16)
A_c   = 0.51
indices["CBCI"] = (1.0 + A_c)*MBSI - A_c*OSAVI

# 11) CBI (eq. 42) with normalised PC1/NDWI/SAVI (eq. 41)
NDWIn = minmax_norm(NDWI)
SAVIn = minmax_norm(SAVI)
indices["CBI"] = safe_div((PC1n + NDWIn - 2.0*SAVIn),
                          (PC1n + NDWIn + 2.0*SAVIn))

# 12) ENDISI (eqs. 21–22)
ratio_s1s2 = safe_div(rho_swir1, rho_swir2)
alpha = 2.0*np.nanmean(rho_blue) / (np.nanmean(ratio_s1s2) + np.nanmean(MNDWI**2) + 1e-6)
ENDISI_num = rho_blue - alpha*(ratio_s1s2 + (MNDWI**2))
ENDISI_den = rho_blue + alpha*(ratio_s1s2 + (MNDWI**2))
indices["ENDISI"] = safe_div(ENDISI_num, ENDISI_den)

# 13) MBAI (eq. 44)
indices["MBAI"] = safe_div(rho_nir + 1.57*rho_green + 2.40*rho_swir1, 1.0 + rho_nir)

# 14) NBAI (eq. 16)
indices["NBAI"] = safe_div(rho_swir2 - safe_div(rho_swir1, rho_green),
                           rho_swir2 + safe_div(rho_swir1, rho_green))

# 15) BRBA (eq. 17)
indices["BRBA"] = safe_div(rho_red, rho_swir1)

# 16) NBI (eq. 15)
indices["NBI"] = safe_div(rho_red * rho_swir1, rho_nir)

# 17) PISI (eq. 13)
indices["PISI"] = 0.8192*rho_blue - 0.5735*rho_nir + 0.0750

# 18) RNDSI (eqs. 37–40)
NDSI   = safe_div(rho_swir2 - rho_green, rho_swir2 + rho_green)
NNDSI  = minmax_norm(NDSI)
NTC1   = minmax_norm(TC1)
indices["RNDSI"] = safe_div(NNDSI, NTC1)

# 19) SwiRed (eq. 20)
indices["SwiRed"] = safe_div(rho_swir1 - rho_red, rho_swir1 + rho_red)

# 20) UI (eq. 14)
indices["UI"] = 100.0*(safe_div(rho_swir2 - rho_nir, rho_swir2 + rho_nir) + 1.0)

# 21) VIBI (eq. 32)
indices["VIBI"] = safe_div(NDVI, NDVI + indices["NDBI"])

# 22) VrNIR-BI (eq. 2)
indices["VrNIR_BI"] = safe_div(rho_red - rho_nir, rho_red + rho_nir)

# 23) VgNIR-BI (eq. 3)
indices["VgNIR_BI"] = safe_div(rho_green - rho_nir, rho_green + rho_nir)

# 24) WE-NDBI (eq. 12) — NDBI masked by Red–Green water gate
W_thresh = 0.3  # you can tune if desired
NDBI_RG = safe_div(rho_red - rho_green, rho_red + rho_green)
water_like = (NDBI_RG <= W_thresh)
WE_NDBI = np.copy(indices["NDBI"])
WE_NDBI[water_like] = 0.0
indices["WE_NDBI"] = WE_NDBI

# 25) UBI (eq. 46) via HSV of composite (NIR, SWIR1, c*Red)
c_ubi = 7.0
rgb_comp = np.stack([
    np.clip(rho_nir,  0, 1),
    np.clip(rho_swir1,0, 1),
    np.clip(c_ubi*rho_red,0,1)
], axis=-1).astype(np.float32)
HSV = rgb_to_hsv(rgb_comp)
H_hue = HSV[...,0]
V_val = HSV[...,2]
indices["UBI"] = safe_div(H_hue - V_val, H_hue + V_val)

# (Optional) Keep NDVI/MNDWI available if you still want to inspect them later
# indices['NDVI'] = NDVI
# indices['MNDWI'] = MNDWI


# ======= 3. NDVI Threshold Search & Best Mask =======
ndvi_thresholds = np.arange(0.1, 0.81, 0.05)
best_results = {}
all_results = {}

def best_threshold(idx, truth, n_steps=200):
    v = np.isfinite(idx) & np.isfinite(truth)
    x, y = idx[v].ravel(), truth[v].ravel()
    if x.size == 0:
        return np.nan  # no valid pixels

    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan

    # constant (or near-constant) index -> any threshold equals lo==hi
    if np.allclose(lo, hi):
        return float(lo)

    best_thr, best_iou = float(lo), -1.0
    for thr in np.linspace(lo, hi, n_steps):
        # use >= so the edge case (thr == max) can still produce positives
        y_pred = (x >= thr).astype(np.uint8)
        iou = jaccard_score(y, y_pred, zero_division=0)
        if iou > best_iou:
            best_thr, best_iou = float(thr), float(iou)
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
            except Exception:
                out["AUC"] = np.nan
        else:
            out["AUC"] = np.nan
    else:
        out["AUC"] = np.nan
    return out

for idx_name, idx in indices.items():
    metric_by_thr = []
    finite_idx = np.isfinite(idx)
    for ndvi_thr in ndvi_thresholds:
        thr = best_threshold(idx, gt_mask_resized)
        if not np.isfinite(thr):
            # fallback if index has no dynamic range / all-NaN over the valid mask
            thr = np.nanpercentile(idx[finite_idx], 95) if np.any(finite_idx) else 0.0
        pred = ((idx >= thr) & (ndvi < ndvi_thr)).astype(np.uint8)
        result = metrics(pred, gt_mask_resized, idx_pred_scores=idx)
        result['ndvi_thr'] = ndvi_thr
        result['idx_thr'] = thr
        metric_by_thr.append(result)

    df_thr = pd.DataFrame(metric_by_thr)
    best_i = df_thr['IoU'].idxmax()
    best_results[idx_name] = df_thr.loc[best_i].to_dict()
    all_results[idx_name] = df_thr

# ======= 4. Visualization: 3-panel for Each Index =======
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
    pred = ((idx > idx_thr) & (ndvi < ndvi_thr)).astype(np.uint8)
    
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

# ======= 5. Save Summary Table =======
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
    plt.savefig(os.path.join(outdir, f"{idx_name}_confusion_matrix.png"), dpi=150)
    plt.close()

for idx_name, idx in indices.items():
    best = best_results[idx_name]
    ndvi_thr = best['ndvi_thr']
    idx_thr = best['idx_thr']
    pred = ((idx > idx_thr) & (NDVI < ndvi_thr)).astype(np.uint8)
    y_true = gt_mask_resized.ravel()
    y_pred = pred.ravel()
    plot_confusion_matrix(y_true, y_pred, idx_name, outdir)

print(f"Confusion matrix PNGs saved to {outdir}")

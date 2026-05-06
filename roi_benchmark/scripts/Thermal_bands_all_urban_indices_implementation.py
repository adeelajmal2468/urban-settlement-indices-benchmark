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
outdir       = "urban_index_eval_thermal_all"
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

# ---------- 1. NDVI, MNDWI (unchanged) ----------
NDVI = safe_div(rho_nir - rho_red, rho_nir + rho_red)
MNDWI = safe_div(rho_green - rho_swir1, rho_green + rho_swir1)

# ---------- 2. Emissivity ε (Sun et al., Eq. 57) ----------
Pv = ((NDVI - 0.2) / 0.3)**2
ε = np.where(
        NDVI < 0.2,
        0.979 - 0.035 * rho_red,
        np.where(
            NDVI <= 0.5,
            0.986 + 0.004 * Pv,
            0.990
        )
)
ε = np.clip(ε, 0.85, 0.999)  # keep within physical bounds

print("Band 10 min:", np.nanmin(b10), "max:", np.nanmax(b10))


# ---------- 3. Land‑surface temperature Ts (Eq. 56) ----------
λ = 10.8e-6          # central wavelength (m) for L8 band 10
ρ_const = 1.438e-2   # m·K   (Planck constant combo)
Tb = b10             # band 10 already in Kelvin
Ts = Tb / (1 + (λ * Tb / ρ_const) * np.log(ε))

# ---------- 4. TRUE PCA for bands (as expected in NDBI_OLI) ----------
from sklearn.decomposition import PCA

def first_pc(a, b):
    h, w = a.shape
    X = np.stack([a.ravel(), b.ravel()], axis=1)
    pc1 = PCA(n_components=1).fit_transform(np.nan_to_num(X)).reshape(h, w)
    return pc1.astype(np.float32)

PCA_6_7   = first_pc(rho_swir1, rho_swir2)
PCA_10_11 = first_pc(b10, b11)


# 1. NDISI_VIS (Xu 2010, Visible)
indices["NDISI_VIS"] = safe_div(3*b10 - b2 + b5 + b6, 3*b10 + b2 + b5 + b6 )

# 2. NDISI_WI (Xu 2010, Water Index, NDWI)
ndwi = safe_div(b3 - b5, b3 + b5)
indices["NDISI_NDWI"] = safe_div(3*b10 - ndwi + b5 + b6, 3*b10 + ndwi + b5 + b6 )

# 3. NDISI_MNDWI (Xu 2010, MNDWI)
mndwi = safe_div(b3 - b6, b3 + b6)
indices["NDISI_MNDWI"] = safe_div(3*b10 - mndwi + b5 + b6, 3*b10 + mndwi + b5 + b6 )

# 4. EBBI (As-syakur et al. 2012)
indices["EBBI"] = safe_div(b6 - b5, 10.0 * np.sqrt(b6 + b10) )

# 5. STRed (SwirTirRed)
indices["STRed"] = safe_div(b6 + b4 - b10, b6 + b10 + b4 )

# 8. NDII (Normalized Difference Impervious Index)
indices["NDII"] = safe_div(b4 - b10, b4 + b10 )

# 11. NDVI (for NDVI-thresholding, classic green-ness index)
ndvi = safe_div(b5 - b4, b5 + b4 )

# 9. DBI (Dry Built Up Index)
indices["DBI"] = safe_div(b2 - b10 - ndvi * (b2 + b10), b2 + b10 )

# ---------- 5. Re‑compute target indices ----------
indices['MNDISI'] = safe_div(Ts - (MNDWI + rho_nir + rho_swir1) / 3,Ts + (MNDWI + rho_nir + rho_swir1) / 3)

indices['NDBI_OLI'] = safe_div((PCA_6_7 + PCA_10_11) - rho_nir,(PCA_6_7 + PCA_10_11) + rho_nir)

indices['BAEM'] = indices['NDBI_OLI'] - NDVI - MNDWI

# keep NDVI, MNDWI, etc. in dict if you still need them
indices['NDVI']  = NDVI
indices['MNDWI'] = MNDWI



# ======= 3. NDVI Threshold Search & Best Mask =======
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
        pred = ((idx > thr) & (ndvi < ndvi_thr)).astype(np.uint8)
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

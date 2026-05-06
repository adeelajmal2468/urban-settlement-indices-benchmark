# ROI Benchmark

This folder contains the second implementation of the urban settlement indices benchmark. It is based on a region of interest over Bahria Town Phase 7, Rawalpindi, and uses Landsat-8, VIIRS nighttime lights, and Dynamic World labels.

This part of the repository is used to reproduce the full ROI experiment. Unlike the SeasoNet benchmark, this implementation supports all 46 urban settlement indices because the required multispectral, panchromatic, thermal, nighttime light, and complex input layers are available.

## Folder structure

```text
roi_benchmark/
├── Results/
├── scripts/
├── Complex Urban Settlement Indices Implementation on ROI Image (Bahria Town Phase 7).docx
├── Multispectral Band Urban Settlement Indices Implementation on ROI Image (Bahria Town Phase 7).docx
├── Night Time Light (NTL) Urban Settlement Indices Implementation on ROI Image (Bahria Town Phase 7).docx
├── Panchromatic Urban Settlement Indices Implementation on ROI Image (Bahria Town Phase 7).docx
├── Thermal Band Urban Settlement Indices Implementation on ROI Image (Bahria Town Phase 7).docx
├── README.md
└── requirements.txt
```

## Environment

All Python scripts were run using Python 3.10.16.

The required packages are listed in `requirements.txt`:

```text
pandas==2.2.3
rasterio==1.4.3
matplotlib==3.10.0
numpy==1.26.4
scikit-learn==1.6.1
scikit-image==0.24.0
seaborn==0.13.2
tqdm==4.67.1
scipy==1.10.1
```

Install the requirements with:

```bash
pip install -r requirements.txt
```

## Study area

The ROI is centered on Bahria Town Phase 7, Rawalpindi.

```text
Longitude: 73.11988
Latitude: 33.53081
Buffer: 2000 meters
Date range: 2023-01-01 to 2025-07-01
Cloud threshold: 10 percent
```

The Google Earth Engine scripts use this ROI to export the Landsat-8 image stack, Dynamic World land cover labels, VIIRS nighttime light layers, vegetation layers, and other supporting raster files.

## Data preparation in Google Earth Engine

The first step is to prepare the ROI dataset in Google Earth Engine. A Google Earth Engine account is required and the JavaScript files should be run inside the Google Earth Engine Code Editor.

The main dataset and label mask are created using:

```text
Google Earth Engine Code for Generating Dataset plus Mask of Landsat 8 Image.js
```

This script exports:

```text
Landsat-8 11 band image for the ROI
Dynamic World land cover label raster
```

The exports are saved to Google Drive from Earth Engine and can then be downloaded for local Python processing.

Dynamic World labels are used as the reference land cover map. The Built class is treated as the urban class and all other classes are treated as non urban.

```text
0 Water
1 Trees
2 Grass
3 Flooded Vegetation
4 Crops
5 Shrub and Scrub
6 Built
7 Bare
8 Snow and Ice
```

For binary evaluation:

```text
Urban: Dynamic World class 6
Non urban: all other Dynamic World classes
```

## Additional Earth Engine exports

Some index families need extra raster inputs. These are generated using the additional Google Earth Engine scripts.

For nighttime light indices, run:

```text
GEE_VIIRS_LANDSAT8_BAHRIA.js
```

This script prepares the VIIRS nighttime light layers and Landsat derived layers needed for NTL based indices. The required files include:

```text
VIIRS_NTL_90p_BahriaPhase7_20230101_20250701.tif
L8_EVImax_BahriaPhase7_20230101_20250701.tif
L8_NDVImax_BahriaPhase7_20230101_20250701.tif
VIIRS_NTL_RGB_BahriaPhase7_2023.tif
UBImax_BahriaPhase7_20230101_20250701.tif
```

For complex indices, run the complex Landsat-8 and VIIRS preparation script:

```text
GEE_Complex_landsat8_bahria.js
```

The main files needed for complex indices are:

```text
VIIRS_NTL_90p_BahriaPhase7_20230101_20250701.tif
L8_NDVImax_BahriaPhase7_20230101_20250701.tif
```

The complex workflow also uses Landsat-8 thermal information and VIIRS nighttime light information together with vegetation information.

## Scripts folder

The `scripts/` folder contains the Python files used to compute the urban settlement indices on the exported ROI rasters.

The main scripts are:

```text
Landsat8_Complex_Indices_Implementation.py
Multispectral_Indices_implementation.py
NTL_indices_implementation.py
Panchromatic_indice_RUI_NRUI_bahria.py
Thermal_bands_all_urban_indices_implementation.py
```

Before running the scripts, update the input file paths inside each script according to the local location of the downloaded GeoTIFF files.

## Index groups implemented

This ROI benchmark implements all 46 urban settlement indices from the full catalogue.

### Multispectral indices

The multispectral script implements the 25 optical indices that use visible, near infrared, and shortwave infrared Landsat-8 bands. These include indices such as NDBI, IBI, BAEI, BBI, BCI, CBCI, MBAI, PISI, UI, WE-NDBI, and UBI.

### Panchromatic indices

The panchromatic script implements indices that use the Landsat-8 panchromatic band together with other Landsat bands. These include:

```text
MNDSI
RUI
NRUI
```

### Thermal indices

The thermal script implements indices that use Landsat-8 thermal bands or land surface temperature related information. These include:

```text
NDISI_VIS
NDISI_NDWI
NDISI_MNDWI
EBBI
MNDISI
STRed
NDBI_OLI
BAEM
NDII
DBI
```

### Nighttime light indices

The NTL script implements indices that use VIIRS nighttime light information together with optical or vegetation based features. These include indices such as HSI, VANUI, LISI, NDUI, NUACI, NAISI, and EUBI.

### Complex indices

The complex script implements indices that combine multiple data sources such as Landsat-8, VIIRS nighttime lights, vegetation indices, and thermal information. These include:

```text
MNDISI Liu
VTLI
TVANUI
```

## General workflow

```text
Run the Google Earth Engine export scripts
        ↓
Download the exported GeoTIFF files from Google Drive
        ↓
Place the files in the local data folder
        ↓
Update paths inside the Python scripts
        ↓
Run the index implementation scripts
        ↓
Generate binary urban masks
        ↓
Compare outputs with the Dynamic World Built reference
        ↓
Save metrics and visual outputs in Results
```

## Results folder

The `Results/` folder stores the outputs generated by the ROI scripts. These outputs include metric files and visual results for the implemented indices.

A typical visual result contains:

```text
Landsat-8 RGB image
Dynamic World labels
Predicted urban overlay
```

The Landsat-8 RGB panel shows the input image over the ROI. The Dynamic World panel shows the reference land cover labels. The overlay panel shows the index based urban prediction compared with the built and non built reference.

These figures are useful for visual inspection because the numerical score alone does not always show boundary errors, water confusion, or mixed pixel behavior.

## Reproduction steps

1. Open Google Earth Engine Code Editor.
2. Run the dataset and mask generation JavaScript file.
3. Run the VIIRS and complex preparation scripts if NTL or complex indices are being reproduced.
4. Export the required GeoTIFF files to Google Drive.
5. Download the exported files locally.
6. Create a Python 3.10.16 environment.
7. Install the packages from `requirements.txt`.
8. Update local file paths inside the scripts.
9. Run the required Python implementation scripts from `scripts/`.
10. Check the generated outputs inside `Results/`.

## Notes

This folder is the full index implementation part of the repository. It is different from the SeasoNet benchmark because it is not limited to Sentinel-2 bands.

The ROI setup uses Landsat-8 for multispectral, panchromatic, and thermal information. It uses VIIRS for nighttime light information. Dynamic World is used as the reference label source, with the Built class used as the positive class.

The ROI experiment should be treated as a reproducible case study. It is useful for testing the complete index catalogue under one consistent workflow, but the results are specific to the selected study area and the chosen reference label source.


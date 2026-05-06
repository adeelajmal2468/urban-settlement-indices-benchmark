# Urban Settlement Indices Benchmark

This repository contains the code and supplementary material for benchmarking urban settlement indices across Sentinel-2, Landsat-8, and VIIRS data. The goal of this repository is to make the experiments reproducible and to provide a clear structure for running both benchmark implementations.

The benchmark is sensor aware. Not every index can be calculated from every satellite sensor, so the repository is divided into two main implementations. The SeasoNet benchmark evaluates the Sentinel-2 compatible multispectral indices across five seasons. The ROI benchmark evaluates the full catalogue of 46 urban settlement indices using Landsat-8, VIIRS nighttime lights, and Dynamic World labels.

## Repository structure

```text
urban-settlement-indices-benchmark/
├── roi_benchmark/
│   ├── Results/
│   ├── scripts/
│   ├── README.md
│   └── requirements.txt
│
├── seasonet_benchmark/
│   ├── Results/
│   ├── scripts/
│   ├── README.md
│   └── requirements.txt
│
└── supplementary_material/
    ├── index_catalogue/
    └── metric_tables/
```

## What is included

This repository contains two benchmark workflows.

1. `seasonet_benchmark/`

   This folder contains the SeasoNet Sentinel-2 implementation. It is used to reproduce the multispectral urban settlement index experiments across fall, winter, snow, spring, and summer.

2. `roi_benchmark/`

   This folder contains the Landsat-8 ROI implementation over Bahria Town Phase 7, Rawalpindi. It is used to reproduce the full 46 index experiment, including multispectral, panchromatic, thermal, nighttime light, and complex indices.

3. `supplementary_material/`

   This folder contains the additional reference material for index formulas, implementation notes, Tasseled Cap coefficients, and metric tables.

## Benchmark overview

Urban settlement indices are algebraic combinations of remote sensing bands or products that are used to detect built up and impervious surfaces. The main challenge is that many indices require different sensor inputs. Some only need visible, near infrared, and shortwave infrared bands. Others require panchromatic data, thermal bands, nighttime light products, or multiple data sources.

For this reason, the benchmark is split into two parts:

```text
SeasoNet benchmark
Sentinel-2 only
25 multispectral indices
Five seasonal subsets

ROI benchmark
Landsat-8, VIIRS, and Dynamic World
All 46 indices
One region of interest case study
```

Both workflows follow the same general logic. Index response maps are calculated, converted into binary built up masks, and evaluated against reference masks using pixel level metrics.

## Implementation 1: SeasoNet Benchmark

The `seasonet_benchmark/` folder contains the Sentinel-2 implementation.

SeasoNet is used because it provides Sentinel-2 Level-2A image patches across five seasonal subsets. Since Sentinel-2 does not provide panchromatic, thermal infrared, or nighttime light bands, this part only implements the indices that can be calculated from Sentinel-2 multispectral bands.

The workflow is:

```text
Download SeasoNet
        ↓
Use meta.csv to create filtered seasonal CSV files
        ↓
Keep samples that contain both urban and non urban classes
        ↓
Run the season specific multispectral index scripts
        ↓
Generate binary built up masks
        ↓
Save metrics and visual results in Results
```

The filtered CSV creation script reads the SeasoNet `meta.csv` file, selects a season, and keeps only the patches where both urban and non urban classes are present.

The class grouping used for filtering is:

```text
Urban classes: 1, 2, 3, 4, 5, 6, 10, 11
Non urban classes: 7, 8, 9, 12 to 33
```

The `Results/` folder stores season wise visual outputs. A typical SeasoNet result contains an RGB image, the index result, the ground truth mask, and the final overlay.

For detailed instructions, see:

```text
seasonet_benchmark/README.md
```

## Implementation 2: ROI Benchmark

The `roi_benchmark/` folder contains the Landsat-8 ROI implementation.

This implementation is based on Bahria Town Phase 7, Rawalpindi. It uses Landsat-8 for multispectral, panchromatic, and thermal inputs. It uses VIIRS for nighttime light information. Dynamic World labels are used as the reference land cover source, where the Built class is treated as urban and all other classes are treated as non urban.

The ROI is defined as:

```text
Longitude: 73.11988
Latitude: 33.53081
Buffer: 2000 meters
Date range: 2023-01-01 to 2025-07-01
```

The dataset and supporting rasters are generated in Google Earth Engine and exported to Google Drive. After downloading the exported GeoTIFF files, the Python scripts in `roi_benchmark/scripts/` are used to compute the indices and evaluate the results.

The ROI benchmark implements all 46 urban settlement indices in these groups:

```text
Multispectral indices
Panchromatic indices
Thermal indices
Nighttime light indices
Complex indices
```

The general workflow is:

```text
Run Google Earth Engine export scripts
        ↓
Download GeoTIFF files from Google Drive
        ↓
Update local file paths inside the Python scripts
        ↓
Run the ROI index implementation scripts
        ↓
Generate binary urban masks
        ↓
Compare outputs with Dynamic World Built reference
        ↓
Save metrics and overlays in Results
```

The `Results/` folder stores metric outputs and visual overlays. A typical ROI result contains a Landsat-8 RGB image, Dynamic World labels, and the predicted urban overlay.

For detailed instructions, see:

```text
roi_benchmark/README.md
```

## Supplementary material

The `supplementary_material/` folder contains additional files used to support the benchmark.

```text
supplementary_material/
├── index_catalogue/
└── metric_tables/
```

### `index_catalogue/`

This folder contains PDF files with the full index catalogue. These files include the index formulas, implementation details, sensor requirements, and Tasseled Cap coefficients.

It also includes separate reference material for:

```text
Sentinel-2 Tasseled Cap components
Landsat-8 Tasseled Cap components
```

These values are used in some index calculations where brightness, greenness, and wetness components are needed.

### `metric_tables/`

This folder contains the benchmark result tables.

It includes:

```text
Full SeasoNet seasonal results
Full 46 index ROI results
```

These tables provide the detailed metric values behind the results reported in the paper and the repository outputs.

## Environment

Both benchmark implementations were run using Python 3.10.16.

Each benchmark folder contains its own `requirements.txt`. The package versions are:

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

To install the requirements inside a benchmark folder, run:

```bash
pip install -r requirements.txt
```

## Data sources

The repository uses publicly available remote sensing datasets.

```text
SeasoNet Sentinel-2 Level-2A dataset
Landsat-8 imagery
VIIRS nighttime light data
Dynamic World land cover labels
```

The SeasoNet benchmark uses the downloaded SeasoNet dataset locally. The ROI benchmark uses Google Earth Engine to export Landsat-8, VIIRS, and Dynamic World layers for the study area.

## Evaluation metrics

Both implementations use pixel level evaluation. Built up pixels are treated as the positive class.

The main metrics are:

```text
Accuracy
Precision
Recall
F1 score
Intersection over Union
True positives
True negatives
False positives
False negatives
```

Visual overlays are also saved to help inspect spatial behaviour, boundary errors, water confusion, vegetation confusion, and mixed pixel effects.

## Main results summary

In the SeasoNet benchmark, WE-NDBI is the strongest overall Sentinel-2 multispectral index across the five seasonal subsets. BBI performs best in winter and snow conditions.

In the ROI benchmark, MBAI gives the best overall ROI result. NDISI with MNDWI, STRED, SwiRed, and MNDSI also perform strongly. These results show that index performance depends on the available sensor inputs, the season, the reference data, and the target landscape.

## How to reproduce

For the SeasoNet benchmark:

```text
1. Open seasonet_benchmark/README.md
2. Download and prepare the SeasoNet dataset
3. Create filtered seasonal CSV files
4. Run the season specific scripts
5. Check outputs in seasonet_benchmark/Results
```

For the ROI benchmark:

```text
1. Open roi_benchmark/README.md
2. Run the Google Earth Engine export scripts
3. Download the exported GeoTIFF files
4. Run the ROI Python scripts
5. Check outputs in roi_benchmark/Results
```

## Notes

The SeasoNet benchmark and ROI benchmark should not be compared as identical experiments. They answer different questions.

The SeasoNet benchmark is useful for testing Sentinel-2 compatible indices across seasons. The ROI benchmark is useful for testing the full 46 index catalogue when panchromatic, thermal, nighttime light, and fusion inputs are available.

Dynamic World is used as a reference source in the ROI benchmark. It is useful for reproducibility, but it should be treated as a proxy reference rather than perfect ground truth.

## Citation

If this repository is used, please cite the related paper once the final citation information is available.

```text
Khan, M. A. A., Younas, J., Dengel, A., Ahmed, S., Faraz, M. M., and Malik, M. I.
Benchmarking Urban Settlement Indices Across Sentinel-2, Landsat-8, and VIIRS.
```

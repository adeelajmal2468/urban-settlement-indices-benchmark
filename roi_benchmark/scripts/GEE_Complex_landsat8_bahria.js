/*********************  USER SETTINGS  **************************/
var BUFFER_METERS = 2000;                       // ~2 km radius
var DATE_START   = '2023-01-01';
var DATE_END     = '2025-07-01';
var CLOUD_THRESH = 10;                          // % cloud cover (scene‑wide)
var DRIVE_FOLDER = 'GEE_Exports';               // change if you like
/****************************************************************/

// 1. Build filtered collection (as you already do)
var col = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
    .filterBounds(roi)
    .filterDate(DATE_START, DATE_END)
    .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_THRESH));

// 2. Mask clouds for each image in the collection
function maskClouds(img) {
  var qa = img.select('QA_PIXEL');
  var good = qa.bitwiseAnd(1 << 0).eq(0)
      .and(qa.bitwiseAnd(1 << 1).eq(0))
      .and(qa.bitwiseAnd(1 << 2).eq(0))
      .and(qa.bitwiseAnd(1 << 3).eq(0));
  return img.updateMask(good);
}
var col_masked = col.map(maskClouds);

// 3. Select the 11 bands
var bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11'];

// 4. Get pixelwise max for each band across all images in the collection
var maxBands = col_masked.select(bands).max().clip(roi);

// 5. Export the annual max composite
Export.image.toDrive({
  image: maxBands,
  description: 'L8_BahriaPhase7_AnnualMax11Bands',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'L8_BahriaPhase7_AnnualMax11Bands_' + DATE_START.replace(/-/g,'') + '_' + DATE_END.replace(/-/g,''),
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// 6. QUICK URBAN MASK  (NDBI>0 & NDVI<0.2)
var ndvi = l8.normalizedDifference(['B5','B4']).rename('NDVI');
var ndbi = l8.normalizedDifference(['B6','B5']).rename('NDBI');
var urbanMask = ndbi.gt(0).and(ndvi.lt(0.2))
                      .rename('urban_mask');

// 7. VIIRS Nighttime Lights extraction (median composite for time range)
var viirs_col = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
                  .filterBounds(roi)
                  .filterDate(DATE_START, DATE_END)
                  .select('avg_rad');

// 2. Create a per-pixel max composite (annual max)
var viirs_ntl_max = viirs_col.max().clip(roi);

// 3. Optionally, 90th percentile instead of strict max:
var viirs_ntl_p90 = viirs_col.reduce(ee.Reducer.percentile([90])).select('avg_rad_p90').clip(roi);

// 4. Export as before
Export.image.toDrive({
  image: viirs_ntl_max,
  description: 'VIIRS_NTL_MAX_BahriaPhase7',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'VIIRS_NTL_MAX_BahriaPhase7_' + DATE_START.replace(/-/g,'') + '_' + DATE_END.replace(/-/g,''),
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

Export.image.toDrive({
  image: viirs_ntl_p90,
  description: 'VIIRS_NTL_90p_BahriaPhase7',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'VIIRS_NTL_90p_BahriaPhase7_' + DATE_START.replace(/-/g,'') + '_' + DATE_END.replace(/-/g,''),
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});


// 8. VISUAL PREVIEW
Map.centerObject(roi, 14);
Map.addLayer(allBands, {bands:['B4','B3','B2'], min:0.05, max:0.4}, 'L8 RGB');
Map.addLayer(urbanMask.updateMask(urbanMask), {palette:['black','red']}, 'Urban mask');
Map.addLayer(viirs_ntl_max, {min:0, max:50, palette:['black','blue','cyan','yellow','white']}, 'NTL (VIIRS)');
Map.addLayer(viirs_ntl_p90,{min:0, max:50, palette:['yellow']}, 'NTL (90 percent)');

// 9. EXPORT — 11‑band image (Landsat)
Export.image.toDrive({
  image: allBands,
  description: 'L8_BahriaPhase7_11Bands',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'L8_BahriaPhase7_' + l8.date().format('yyyyMMdd').getInfo(),
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// 10. EXPORT — binary urban mask (Landsat-based)
Export.image.toDrive({
  image: urbanMask,
  description: 'UrbanMask_BahriaPhase7',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'UrbanMask_BahriaPhase7_' + l8.date().format('yyyyMMdd').getInfo(),
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// ******************************
//      USER SETTINGS
// ******************************
var BUFFER_METERS = 2000; // ~2 km radius
var DATE_START   = '2023-01-01';
var DATE_END     = '2025-07-01';
var CLOUD_THRESH = 10;    // % cloud cover
var DRIVE_FOLDER = 'GEE_Exports';
// ROI: Bahria Town Phase-7, Rawalpindi
var roi = ee.Geometry.Point(73.11988, 33.53081).buffer(BUFFER_METERS);

// ******************************
//      IMAGE COLLECTION
// ******************************
var col = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
    .filterBounds(roi)
    .filterDate(DATE_START, DATE_END)
    .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_THRESH));

// ******************************
//      NDVI CALCULATION
// ******************************
var addNDVI = function(img) {
  var ndvi = img.normalizedDifference(['B5', 'B4']).rename('NDVI');
  return img.addBands(ndvi);
};
var col_ndvi = col.map(addNDVI);

// ******************************
//      MAX COMPOSITES
// ******************************
var ndvi_max = col_ndvi.select('NDVI').max().clip(roi);

// ******************************
//      EXPORT TO DRIVE
// ******************************

Export.image.toDrive({
  image: ndvi_max,
  description: 'L8_NDVImax_BahriaPhase7',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'L8_NDVImax_BahriaPhase7_' + DATE_START.replace(/-/g,'') + '_' + DATE_END.replace(/-/g,''),
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

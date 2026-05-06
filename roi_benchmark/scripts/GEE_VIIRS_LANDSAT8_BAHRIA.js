/*********************  USER SETTINGS  **************************/
var BUFFER_METERS = 2000;                       // ~2 km radius
var DATE_START   = '2023-01-01';
var DATE_END     = '2025-07-01';
var CLOUD_THRESH = 10;                          // % cloud cover (scene‑wide)
var DRIVE_FOLDER = 'GEE_Exports';               // change if you like
/****************************************************************/

// 1. ROI: Bahria Town Phase‑7, Rawalpindi
var roi = ee.Geometry.Point(73.11988, 33.53081).buffer(BUFFER_METERS);

// 2. Landsat‑8 TOA collection (includes PAN + TIRS 2)
var col = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
            .filterBounds(roi)
            .filterDate(DATE_START, DATE_END)
            .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_THRESH))
            .sort('CLOUD_COVER');

// 3. Pick the **first** scene after sorting (lowest cloud)
var l8 = ee.Image(col.first());

// 4. QA‑based valid‑pixel mask  (remove cloud / cirrus / fill)
var qa   = l8.select('QA_PIXEL');
var good = qa.bitwiseAnd(1 << 0).eq(0)   // fill
            .and(qa.bitwiseAnd(1 << 1).eq(0)) // dilated cloud
            .and(qa.bitwiseAnd(1 << 2).eq(0)) // cirrus
            .and(qa.bitwiseAnd(1 << 3).eq(0)); // cloud
l8 = l8.updateMask(good).clip(roi);

// 5. ALL 11 BANDS  (B1…B11)
var allBands = l8.select([
  'B1','B2','B3','B4','B5','B6','B7',
  'B8',      // PAN (15 m)
  'B9',      // Cirrus
  'B10','B11'// TIRS 1 & 2
]);

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
//      EVI CALCULATION
// ******************************
var addEVI = function(img) {
  var evi = img.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {
      'NIR': img.select('B5'),
      'RED': img.select('B4'),
      'BLUE': img.select('B2')
    }
  ).rename('EVI');
  return img.addBands(evi);
};
var col_evi = col.map(addEVI);

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
var evi_max = col_evi.select('EVI').max().clip(roi);
var ndvi_max = col_ndvi.select('NDVI').max().clip(roi);

// ******************************
//      EXPORT TO DRIVE
// ******************************
Export.image.toDrive({
  image: evi_max,
  description: 'L8_EVImax_BahriaPhase7',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'L8_EVImax_BahriaPhase7_' + DATE_START.replace(/-/g,'') + '_' + DATE_END.replace(/-/g,''),
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

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

var months = [
  {'name': 'Jan', 'start': '2023-01-01', 'end': '2023-01-31'},
  {'name': 'May', 'start': '2023-05-01', 'end': '2023-05-31'},
  {'name': 'Sep', 'start': '2023-09-01', 'end': '2023-09-30'}
];

var roi = ee.Geometry.Point(73.11988, 33.53081).buffer(2000);

var ntl_bands = [];
months.forEach(function(m) {
  var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
    .filterBounds(roi)
    .filterDate(m.start, m.end)
    .select('avg_rad')
    .mean()
    .clip(roi)
    .rename(m.name);
  ntl_bands.push(viirs);
});

// Stack into one image with 3 bands (Jan, May, Sep)
var ntl_rgb = ntl_bands[0].addBands(ntl_bands[1]).addBands(ntl_bands[2]);

Export.image.toDrive({
  image: ntl_rgb,
  description: 'VIIRS_NTL_RGB_BahriaPhase7',
  folder: 'GEE_Exports',
  fileNamePrefix: 'VIIRS_NTL_RGB_BahriaPhase7_2023',
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// User settings (as before)
var roi = ee.Geometry.Point(73.11988, 33.53081).buffer(2000);
var DATE_START = '2023-01-01';
var DATE_END   = '2025-07-01';

// Function to compute UBI for one Landsat 8 image
var addUBI = function(img) {
  // UBI synthetic RGB (NIR, SWIR1, 2*Red)
  var nir = img.select('B5');
  var swir1 = img.select('B6');
  var red = img.select('B4').multiply(2);
  var rgb = ee.Image.cat(nir, swir1, red).rename(['nir', 'swir1', '2red']);
  // Normalize each band to 0–1 within the image
  var rgb_min = rgb.reduceRegion(ee.Reducer.min(), roi, 30);
  var rgb_max = rgb.reduceRegion(ee.Reducer.max(), roi, 30);
  var norm = rgb.subtract(ee.Image.constant([rgb_min.get('nir'), rgb_min.get('swir1'), rgb_min.get('2red')]))
                .divide(ee.Image.constant([rgb_max.get('nir'), rgb_max.get('swir1'), rgb_max.get('2red')])
                        .subtract(ee.Image.constant([rgb_min.get('nir'), rgb_min.get('swir1'), rgb_min.get('2red')])).add(1e-6));
  // Convert to HSV, extract H and V
  var hsv = norm.rgbToHsv();
  var H = hsv.select('hue');
  var V = hsv.select('value');
  // UBI formula
  var ubi = H.subtract(V).divide(H.add(V).add(1e-6)).rename('UBI');
  return img.addBands(ubi);
};

// Build collection, add UBI, mask clouds as needed
var col = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
    .filterBounds(roi)
    .filterDate(DATE_START, DATE_END)
    .map(addUBI);

var ubi_col = col.select('UBI');

// Compute pixelwise max
var ubi_max = ubi_col.max().clip(roi);

// Export to Drive
Export.image.toDrive({
  image: ubi_max,
  description: 'UBImax_BahriaPhase7',
  folder: 'GEE_Exports',
  fileNamePrefix: 'UBImax_BahriaPhase7_' + DATE_START.replace(/-/g,'') + '_' + DATE_END.replace(/-/g,''),
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});


// 13. OPTIONAL: print the chosen Landsat scene ID
print('Scene chosen:', l8.id());

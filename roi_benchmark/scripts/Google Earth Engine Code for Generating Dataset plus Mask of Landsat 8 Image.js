/*********************  USER SETTINGS  **************************/
var BUFFER_METERS = 2000;                       // ~2 km radius
var DATE_START   = '2023-01-01';
var DATE_END     = '2025-07-01';
var CLOUD_THRESH = 10;                          // % cloud cover (scene‑wide)
var DRIVE_FOLDER = 'GEE_Exports';               // change if you like
/****************************************************************/

// 1. ROI: Bahria Town Phase‑7, Rawalpindi
var roi = ee.Geometry.Point(73.11988, 33.53081).buffer(BUFFER_METERS);

// 2.  Landsat‑8 TOA collection (includes PAN + TIRS 2)
var col = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
            .filterBounds(roi)
            .filterDate(DATE_START, DATE_END)
            .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_THRESH))
            .sort('CLOUD_COVER');

// 3. Pick the **first** scene after sorting (lowest cloud)
var l8 = ee.Image(col.first());

// 4.  QA‑based valid‑pixel mask  (remove cloud / cirrus / fill)
var qa   = l8.select('QA_PIXEL');
var good = qa.bitwiseAnd(1 << 0).eq(0)   // fill
           .and(qa.bitwiseAnd(1 << 1).eq(0)) // dilated cloud
           .and(qa.bitwiseAnd(1 << 2).eq(0)) // cirrus
           .and(qa.bitwiseAnd(1 << 3).eq(0)); // cloud
l8 = l8.updateMask(good).clip(roi);

// 5.  ALL 11 BANDS  (B1…B11)
var allBands = l8.select([
  'B1','B2','B3','B4','B5','B6','B7',
  'B8',      // PAN (15 m)
  'B9',      // Cirrus
  'B10','B11'// TIRS 1 & 2
]);

// 6.  QUICK URBAN MASK  (NDBI>0 & NDVI<0.2)
var ndvi = l8.normalizedDifference(['B5','B4']).rename('NDVI');
var ndbi = l8.normalizedDifference(['B6','B5']).rename('NDBI');
var urbanMask = ndbi.gt(0).and(ndvi.lt(0.2))
                     .rename('urban_mask');

// 7.  VISUAL PREVIEW  — make sure you see data!
Map.centerObject(roi, 14);
Map.addLayer(allBands,           // RGB preview
             {bands:['B4','B3','B2'], min:0.05, max:0.4},
             'L8 RGB');
Map.addLayer(urbanMask.updateMask(urbanMask),
             {palette:['black','red']},
             'Urban mask');

// 8.  EXPORT  — 11‑band image
Export.image.toDrive({
  image: allBands,
  description: 'L8_BahriaPhase7_11Bands',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'L8_BahriaPhase7_'+l8.date().format('yyyyMMdd').getInfo(),
  region: roi,
  scale: 30,                     // uniform 30 m
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// 9.  EXPORT  — binary urban mask
Export.image.toDrive({
  image: urbanMask,
  description: 'UrbanMask_BahriaPhase7',
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'UrbanMask_BahriaPhase7_'+l8.date().format('yyyyMMdd').getInfo(),
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// 10.  OPTIONAL: print the chosen scene ID for your records
print('Scene chosen:', l8.id());

//1. Define ROI
var roi = ee.Geometry.Point(73.11988, 33.53081).buffer(2000);
Map.centerObject(roi, 14);

// 2. Load Dynamic World (Jan 2023 - Jun 2025)
var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(roi)
  .filterDate('2023-01-01', '2025-06-30');

// 3. Landcover class mode (per pixel: most common label)
var dwLabels = dw.select('label').mode();

// 4. (Optional) Visualize: color palette for Dynamic World
var dwPalette = [
  '#419bdf', // 0 Water
  '#397d49', // 1 Trees
  '#88b053', // 2 Grass
  '#7a87c6', // 3 Flooded Vegetation
  '#e49635', // 4 Crops
  '#dfc35a', // 5 Shrub & Scrub
  '#c4281b', // 6 Built
  '#a59b8f', // 7 Bare
  '#ffffff'  // 8 Snow & Ice
];
Map.addLayer(
  dwLabels.clip(roi),
  {min: 0, max: 8, palette: dwPalette},
  'Dynamic World Landcover (mode)'
);

// 5. Landsat-8 TOA, least-cloudy scene
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
  .filterBounds(roi)
  .filterDate('2023-01-01', '2025-06-30')
  .sort('CLOUD_COVER')
  .first();

// 6. Select all 11 bands (B1 to B11)
var allBands = l8.select([
  'B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11'
]);

// 7. (Optional) Visualize Landsat-8 RGB
Map.addLayer(
  allBands.clip(roi),
  {bands: ['B4','B3','B2'], min: 0.05, max: 0.4},
  'Landsat-8 RGB'
);

// 8. EXPORT: Full 11-band Landsat-8 image
Export.image.toDrive({
  image: allBands.clip(roi),
  description: 'L8_BahriaPhase7_11Bands',
  folder: 'GEE_Exports',
  fileNamePrefix: 'L8_BahriaPhase7_11Bands_' + l8.date().format('yyyyMMdd').getInfo(),
  region: roi,
  scale: 30, // Landsat-8 native resolution
  crs: 'EPSG:4326',
  maxPixels: 1e10
});

// 9. EXPORT: Dynamic World label raster (all classes)
Export.image.toDrive({
  image: dwLabels.clip(roi),
  description: 'DynamicWorld_LandcoverLabels',
  folder: 'GEE_Exports',
  fileNamePrefix: 'DynamicWorld_LandcoverLabels',
  region: roi,
  scale: 10, // Dynamic World native resolution
  crs: 'EPSG:4326',
  maxPixels: 1e10
});


// // --------- 0. User Settings ---------
// var buffer_m      = 4000;    // Patch radius in meters
// var spacing_deg   = 0.22;    // Grid step in degrees (~22 km; make larger if memory issues)
// var min_fraction  = 0.10;    // Min % for a class to "count"
// var min_classes   = 7;       // Require at least 7 classes present

// // --------- 1. Define 6 Pakistan Tiles ---------
// // [xmin, ymin, xmax, ymax]
// var regions = [
//   ee.Geometry.Rectangle([66, 23.5, 69.5, 26.5]),   // Sindh/South
//   ee.Geometry.Rectangle([69.5, 23.5, 73, 28]),     // SE Punjab/Balochistan
//   ee.Geometry.Rectangle([73, 23.5, 76.5, 28]),     // East Punjab/South KP
//   ee.Geometry.Rectangle([66, 26.5, 70, 31.5]),     // Central Pak/Balochistan/West Punjab
//   ee.Geometry.Rectangle([70, 28, 75, 34]),         // North Punjab/KP/AJK
//   ee.Geometry.Rectangle([72.5, 34.2, 77, 37.1])    // Extreme North, Gilgit-Baltistan (Snow)
// ];
// var region_names = [
//   'Sindh_South', 'SE_Punjab_Baloch', 'East_Punjab_S_KP', 'Central_Pak', 'N_Punjab_KP_AJK', 'North_Snow'
// ];

// // --------- 2. Choose which tile to run (edit this index) ---------
// var REGION_IDX = 5; // <--- Change 0..5 to run different region each time!
// var region     = regions[REGION_IDX];
// var region_tag = region_names[REGION_IDX];

// // --------- 3. Grid Generator (fixed robust version) ---------
// function makeGrid(region, step) {
//   var ring = ee.List(region.bounds().coordinates().get(0));
//   var xmin = ee.Number(ee.List(ring.get(0)).get(0));
//   var ymin = ee.Number(ee.List(ring.get(0)).get(1));
//   var xmax = ee.Number(ee.List(ring.get(2)).get(0));
//   var ymax = ee.Number(ee.List(ring.get(2)).get(1));
//   var lons = ee.List.sequence(xmin, xmax, step);
//   var lats = ee.List.sequence(ymin, ymax, step);
//   var pts = lats.map(function(lat){
//     return lons.map(function(lon){
//       return ee.Feature(ee.Geometry.Point([lon, lat]));
//     });
//   }).flatten();
//   return ee.FeatureCollection(pts);
// }

// // --------- 4. Dynamic World Label Image ---------
// var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
//   .filterDate('2023-01-01', '2025-06-30')
//   .select('label')
//   .mode();

// // --------- 5. Grid for This Region ---------
// var grid = makeGrid(region, spacing_deg);

// // --------- 6. Compute Class Counts ---------
// var checked = grid.map(function(pt){
//   var geom = ee.Feature(pt).geometry().buffer(buffer_m);
//   var countsDict = dw.reduceRegion({
//     reducer: ee.Reducer.frequencyHistogram(),
//     geometry: geom,
//     scale: 10,
//     maxPixels: 1e8
//   });
//   var counts = ee.Dictionary(countsDict.get('label'));
//   var total = ee.Number(counts.values().reduce(ee.Reducer.sum()));
//   // Classes present above threshold
//   var present = counts.values().filter(ee.Filter.gte('item', total.multiply(min_fraction))).length();

//   return ee.Feature(pt).set({
//     'num_classes': present,
//     'counts': counts,
//     'total_pix': total
//   });
// });

// // --------- 7. Filter for Diverse Sites ---------
// var diverse = checked.filter(ee.Filter.gte('num_classes', min_classes));

// // --------- 8. (Optional) For North Only: Filter for Snow Presence ---------
// if (region_tag === 'North_Snow') {
//   diverse = diverse.map(function(f) {
//     var counts = ee.Dictionary(f.get('counts'));
//     var total = ee.Number(f.get('total_pix'));
//     var snow = ee.Number(counts.get('8', 0)); // If no snow, returns 0
//     var frac = snow.divide(total);
//     return f.set('snow_frac', frac);
//   }).filter(ee.Filter.gte('snow_frac', 0.02)); // Keep if ≥2% snow/ice
// }

// // --------- 9. Preview ---------
// print('Diverse ROIs:', diverse.limit(20));
// Map.centerObject(region, 7);
// Map.addLayer(region, {color: 'blue'}, region_tag);
// Map.addLayer(diverse.limit(100), {color: 'red'}, 'Diverse ROI Grid');

// // --------- 10. Export as CSV (if >0 results) ---------
// Export.table.toDrive({
//   collection: diverse,
//   description: 'PK_Diverse_ROI_'+region_tag,
//   fileFormat: 'CSV'
// });

// // --------- 11. Class Legend ---------
// print('Class codes: 0=Water, 1=Trees, 2=Grass, 3=FloodedVeg, 4=Crops, 5=Shrub, 6=Built, 7=Bare, 8=Snow');

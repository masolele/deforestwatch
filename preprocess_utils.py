import ee
import numpy as np

# Sentinel-1 preprocessing pipeline (with slope correction)
def lin_to_db(image):
    bandNames = image.bandNames().remove('angle')
    db = ee.Image.constant(10).multiply(image.select(bandNames).log10()).rename(bandNames)
    return image.addBands(db, None, True)

def maskAngGT30(image):
    ang = image.select(['angle'])
    return image.updateMask(ang.gt(30.63993)).set('system:time_start', image.get('system:time_start'))

def maskAngLT452(image):
    ang = image.select(['angle'])
    return image.updateMask(ang.lt(45.23993)).set('system:time_start', image.get('system:time_start'))

def slope_correction(collection):
    DEM = ee.Image('USGS/SRTMGL1_003')
    ninetyRad = ee.Image.constant(90).multiply(np.pi/180)

    def _volumetric_model_SCF(theta_iRad, alpha_rRad):
        nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
        denominator = (ninetyRad.subtract(theta_iRad)).tan()
        return nominator.divide(denominator)

    def _masking(alpha_rRad, theta_iRad, buffer=0):
        layover = alpha_rRad.lt(theta_iRad)
        shadow = alpha_rRad.gt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad)))
        mask = layover.And(shadow)
        return mask.rename('no_data_mask')

    def _correct(image):
        theta_iRad = image.select('angle').multiply(np.pi / 180)
        elevation = DEM.resample('bilinear').clip(image.geometry())
        alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(np.pi / 180)
        phi_iRad = ee.Image.constant(0)  # Assumes mean heading = 0
        phi_sRad = ee.Terrain.aspect(elevation).multiply(np.pi / 180)
        phi_rRad = phi_iRad.subtract(phi_sRad)
        alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()
        gamma0 = image.divide(theta_iRad.cos())
        scf = _volumetric_model_SCF(theta_iRad, alpha_rRad)
        gamma0_flat = gamma0.multiply(scf)
        mask = _masking(alpha_rRad, theta_iRad)
        return gamma0_flat.mask(mask).copyProperties(image, ["system:time_start"]).addBands(image.select('angle'))

    return collection.map(_correct)

def preproc_s1(s1_collection):
    s1_collection = slope_correction(s1_collection)
    s1_collection = s1_collection.map(maskAngGT30)
    s1_collection = s1_collection.map(maskAngLT452)
    s1_collection = s1_collection.map(lin_to_db)
    return s1_collection

# Create S1 composite
def create_s1_composite(roi, start_date, end_date, method='median'):
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
    
    s1_preproc = preproc_s1(s1).select('VV', 'VH')
    if method == 'median':
        return s1_preproc.median()
    elif method == 'mean':
        return s1_preproc.mean()
    else:
        return s1_preproc.sort('system:time_start', False).mosaic()

# Normalize helper functions
def normalize_band(array, min_val, max_val):
    array = np.clip(array, min_val, max_val)
    return (array - min_val) / (max_val - min_val)

# Full preprocessing pipeline
def preprocess_planet(roi):
    START_DATE = '2024-01-01'
    END_DATE = '2024-12-30'

    # Sentinel-2
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(START_DATE, END_DATE) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median() \
        .clip(roi) \
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12'])

    # Sentinel-1
    s1 = create_s1_composite(roi, START_DATE, END_DATE)

    # Elevation, Lon, Lat
    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clip(roi)
    lonlat = ee.Image.pixelLonLat().clip(roi)
    lon = lonlat.select('longitude')
    lat = lonlat.select('latitude')

    # NDVI, NDRE, EVI
    nir = s2.select('B8')
    red = s2.select('B4')
    green = s2.select('B3')
    red_edge1 = s2.select('B5')

    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    evi = nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(6)).subtract(green.multiply(7.5)).add(1)).rename('EVI')
    ndre = nir.subtract(red_edge1).divide(nir.add(red_edge1)).rename('NDRE')

    # Forest loss mask
    hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
    loss = hansen.select('loss').clip(roi)

    # Stack everything
    full_image = s2 \
        .addBands(s1) \
        .addBands(ndvi) \
        .addBands(ndre) \
        .addBands(evi) \
        .addBands(dem.rename('elevation')) \
        .addBands(lon) \
        .addBands(lat) \
        .addBands(loss)

    # Download thumbnail to NumPy
    url = full_image.getThumbURL({
        'region': roi.bounds().getInfo()['coordinates'],
        'dimensions': 256,
        'format': 'png',
        'min': 0,
        'max': 3000
    })

    import requests
    from PIL import Image
    from io import BytesIO

    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.nan_to_num(arr)
    return arr

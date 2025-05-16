import ee
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# S1 preprocessing steps
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
    ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)

    def _volumetric_model_SCF(theta_iRad, alpha_rRad):
        nom = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
        denom = (ninetyRad.subtract(theta_iRad)).tan()
        return nom.divide(denom)

    def _masking(alpha_rRad, theta_iRad):
        layover = alpha_rRad.lt(theta_iRad)
        shadow = alpha_rRad.gt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad)))
        return layover.And(shadow).rename('no_data_mask')

    def _correct(image):
        try:
            theta_iRad = image.select('angle').multiply(np.pi / 180)
            elevation = DEM.resample('bilinear').clip(image.geometry())
            alpha_sRad = ee.Terrain.slope(elevation).multiply(np.pi / 180)
            phi_iRad = ee.Image.constant(0)
            phi_sRad = ee.Terrain.aspect(elevation).multiply(np.pi / 180)
            phi_rRad = phi_iRad.subtract(phi_sRad)
            alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()
            gamma0 = image.divide(theta_iRad.cos())
            scf = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan().divide(
                (ninetyRad.subtract(theta_iRad)).tan()
            )
            gamma0_flat = gamma0.multiply(scf)
            mask = alpha_rRad.lt(theta_iRad).And(
                alpha_rRad.gt(ee.Image.constant(-1).multiply(ninetyRad.subtract(theta_iRad)))
            )
    
            # Explicit cast to ee.Image at each step
            gamma0_masked = ee.Image(gamma0_flat.mask(mask))
            angle_band = image.select('angle')
            result = gamma0_masked.addBands(angle_band)
            return result.copyProperties(image, ["system:time_start"])
        
        except Exception as e:
            print("❌ Error inside _correct():", str(e))
            return ee.Image.constant(0).rename(['VV', 'VH', 'angle'])  # failsafe image

    return collection.map(_correct)

def preproc_s1(s1_collection):
    return slope_correction(s1_collection) \
        .map(maskAngGT30) \
        .map(maskAngLT452) \
        .map(lin_to_db)

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

# Main preprocessing function: returns 17-band NumPy array
def preprocess_planet(roi, start_date, end_date):
    import requests
    from PIL import Image
    from io import BytesIO

    # Sentinel-2 Bands
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median() \
        .clip(roi)

    s2 = s2.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A'])

    nir = s2.select('B8')
    red = s2.select('B4')
    green = s2.select('B3')
    red_edge1 = s2.select('B5')

    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    evi = nir.subtract(red).multiply(2.5).divide(
        nir.add(red.multiply(6)).subtract(green.multiply(7.5)).add(1)).rename('EVI')
    evi = evi.clamp(-1, 1).add(1).divide(2)
    ndre = nir.subtract(red_edge1).divide(nir.add(red_edge1)).rename('NDRE')
    ndre = ndre.clamp(-1, 1).add(1).divide(2)

    # Sentinel-1 VV and VH (normalized)
    s1 = create_s1_composite(roi, start_date, end_date)
    vv = s1.select('VV').clamp(-25, 0).add(25).divide(25).rename('VV_norm')
    vh = s1.select('VH').clamp(-30, -5).add(30).divide(25).rename('VH_norm')

    # Elevation
    elevation = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clamp(-400, 8000).add(400).divide(8400).clip(roi)

    # Latitude / Longitude (normalized)
    lonlat = ee.Image.pixelLonLat().clip(roi)
    lon = lonlat.select('longitude').add(180).divide(360).rename('lon_norm')
    lat = lonlat.select('latitude').add(60).divide(120).rename('lat_norm')

    # Stack all 17 bands
    image = s2 \
        .addBands(ndvi) \
        .addBands(ndre) \
        .addBands(evi) \
        .addBands(vv) \
        .addBands(vh) \
        .addBands(elevation.rename('elevation')) \
        .addBands(lon) \
        .addBands(lat)

    # Export to thumbnail (low-res for demo/testing)
    url = image.getThumbURL({
        'region': roi.bounds().getInfo()['coordinates'],
        'dimensions': 256,
        'format': 'png',
        'min': 0,
        'max': 1
    })

    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.repeat(arr[:, :, np.newaxis], 17, axis=2)  # TEMP: simulate 17-band shape

    return np.nan_to_num(arr)

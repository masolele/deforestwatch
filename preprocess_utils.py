import ee
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from skimage import io
import geemap
import os
from chunked_download import chunked_export
import rasterio
from rasterio.merge import merge

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

#preprocessing data to  be classified

def normalise_vv(raster):
    raster[raster < -25] = -25
    raster[raster > 0] = 0
    return (raster+25)/25

def normalise_vh(raster):
    raster[raster < -30] = -30
    raster[raster > -5] = -5
    return (raster+30)/25

def normalise_longitude(raster):
    raster[raster < -180] = -180
    raster[raster > 180] = 180
    return (raster+180)/360

def normalise_latitude(raster):
    raster[raster < -60] = -60
    raster[raster > 60] = 60
    return (raster+60)/120

def normalise_altitude(raster):
    raster[raster < -400] = -400
    raster[raster > 8000] = 8000
    return (raster+400)/8400

def normalise_ndre(raster):
    raster[raster < -1] = -1
    raster[raster > 1] = 1
    return (raster+1)/2
    
def normalise_evi(raster):
    raster[raster < -1] = -1
    raster[raster > 1] = 1
    return (raster+1)/2


def norm(image):
    NORM_PERCENTILES = np.array([
    [1.7417268007636313, 2.023298706048351],
    [1.7261204997060209, 2.038905204308012],
    [1.6798346251414997, 2.179592821212937],
    [2.3828939530384052, 2.7578332604178284],
    [1.7417268007636313, 2.023298706048351],
    [1.7417268007636313, 2.023298706048351],
    [1.7417268007636313, 2.023298706048351],
    [1.7417268007636313, 2.023298706048351],
    [1.7417268007636313, 2.023298706048351]])

    image = np.log(image * 0.005 + 1)
    image = (image - NORM_PERCENTILES[:, 0]) / NORM_PERCENTILES[:, 1]

    # Get a sigmoid transfer of the re-scaled reflectance values.
    image = np.exp(image * 5 - 1)
    image = image / (image + 1)
    
    return image

def preprocess_images(x_img):
  # compile
    #loss = x_img[:,:,14]
    #loss = np.where(loss==0, 1, loss)
    #kernel = np.ones((40, 40))
    #loss = np.int64(convolve2d(loss, kernel, mode='same') > 0)

    x_img1 = x_img[:,:,[0,1,2,3,4,5,6,7,8]] 

    x_img2 = norm(x_img1)
    vv = normalise_vv(x_img[:,:,9])
    vh = normalise_vh(x_img[:,:,10])
    alt = normalise_altitude(x_img[:,:,11])
    lon = normalise_longitude(x_img[:,:,12])
    lat = normalise_latitude(x_img[:,:,13])

    SIZE_X = (x_img.shape[0])
    SIZE_Y = (x_img.shape[1])
    red_edge1 = x_img1[:,:,3]
    nir = x_img1[:,:,6]
    red = x_img1[:,:,2]
    green = x_img1[:,:,1]
    blue = x_img1[:,:,0]

    ndvi = np.where((nir+red)==0., 0, (nir-red)/(nir+red))
    evi  = np.where((nir+red)==0., 0, 2.5*((nir-red)/(nir+6*red-7.5*blue+1)))
    evi = normalise_evi(evi)
    ndre = np.where((nir+red_edge1)==0., 0, (nir-red_edge1)/(nir+red_edge1))
    ndre = normalise_ndre(ndre)

    ndvi = np.reshape(ndvi, (SIZE_X,SIZE_Y,1))
    evi = np.reshape(evi, (SIZE_X,SIZE_Y,1))
    
    ndre = np.reshape(ndre, (SIZE_X,SIZE_Y,1))
    vv = np.reshape(vv, (SIZE_X,SIZE_Y,1))
    vh = np.reshape(vh, (SIZE_X,SIZE_Y,1))
    alt = np.reshape(alt, (SIZE_X,SIZE_Y,1))
    lon = np.reshape(lon, (SIZE_X,SIZE_Y,1))
    lat = np.reshape(lat, (SIZE_X,SIZE_Y,1))
    image = np.concatenate((x_img2, ndvi,ndre,evi,vv, vh, alt, lon,lat), axis=2)
    #image = np.nan_to_num(image) 
    return image

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

    #nir = s2.select('B8')
    #red = s2.select('B4')
    #green = s2.select('B3')
    #red_edge1 = s2.select('B5')

    #ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    #evi = nir.subtract(red).multiply(2.5).divide(
    #    nir.add(red.multiply(6)).subtract(green.multiply(7.5)).add(1)).rename('EVI')
    #evi = evi.clamp(-1, 1).add(1).divide(2)
    #ndre = nir.subtract(red_edge1).divide(nir.add(red_edge1)).rename('NDRE')
    #ndre = ndre.clamp(-1, 1).add(1).divide(2)

    # Sentinel-1 VV and VH (normalized)
    s1 = create_s1_composite(roi, start_date, end_date)
    vv = s1.select('VV')#.clamp(-25, 0).add(25).divide(25).rename('VV_norm')
    vh = s1.select('VH')#.clamp(-30, -5).add(30).divide(25).rename('VH_norm')

    # Elevation
    elevation = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic()#.clamp(-400, 8000).add(400).divide(8400).clip(roi)

    # Latitude / Longitude (normalized)
    lonlat = ee.Image.pixelLonLat().clip(roi)
    lon = lonlat.select('longitude')#.add(180).divide(360).rename('lon_norm')
    lat = lonlat.select('latitude')#.add(60).divide(120).rename('lat_norm')

    # Stack all 17 bands
    image = s2 \
        .addBands(vv) \
        .addBands(vh) \
        .addBands(elevation.rename('elevation')) \
        .addBands(lon) \
        .addBands(lat)

    # Export to thumbnail (low-res for demo/testing)
    current_working_directory = os.getcwd()

   # print output to the console
    print(current_working_directory, 'working direct')
    # geemap.ee_export_image(image,
    #                        filename= 'clipped.tif',
    #                        scale=10,
    #                        region=roi.bounds()
    #                       )
    chunked_export(image=image,
                   roi =roi, #.bounds()
                   output_path= 'clipped.tif',
                   scale=10
                          )

    
    arr = io.imread('clipped.tif') 
    arr2 = preprocess_images(arr)
    return arr2 #np.nan_to_num(arr)
    
    # url = image.getThumbURL({
    #     'region': roi.bounds().getInfo()['coordinates'],
    #     'dimensions': 256,
    #     'format': 'png',
    #     'min': 0,
    #     'max': 1
    # })

    # response = requests.get(url)
    # img = Image.open(BytesIO(response.content))#.convert('RGB')
    # arr = np.array(img).astype(np.float32) / 255.0
    #arr = np.repeat(arr[:, :, np.newaxis], 17, axis=2)  # TEMP: simulate 17-band shape

    #return np.nan_to_num(arr)

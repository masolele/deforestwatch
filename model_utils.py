import ee
from keras.models import load_model

regions = {
    'Africa': ee.Geometry.Rectangle([-20.0, -35.0, 55.0, 40.0]),
    'Asia': ee.Geometry.Rectangle([55.0, -10.0, 150.0, 60.0]),
    'Latin America': ee.Geometry.Rectangle([-95.0, -55.0, -30.0, 20.0])
}

region_models = {
    'Africa': 'models/sentAfrica.hdf5',
    'Asia': 'models/sentAsia.hdf5',
    'Latin America': 'models/sentLatinAmerica.hdf5'
}

def get_region_from_roi(roi):
    centroid = roi.centroid()
    for name, geom in regions.items():
        if geom.contains(centroid).getInfo():
            return name
    return None

def load_region_model(region_name):
    return load_model(region_models[region_name], compile=False)

import os
import ee
from keras.models import load_model
from huggingface_hub import hf_hub_download

# Region-to-model file names
region_models = {
    'Africa': 'best_weights_att_unet_lagtime_5_Fused3_2023_totalLoss6V1_without_loss_sentAfrica6.hdf5',
    'Asia': 'best_weights_VIT_FusionSEA.hdf5',
    'Latin America': 'best_weights_VIT_FusionSA1.hdf5'
}

# Bounding boxes for region detection
regions = {
    'Africa': ee.Geometry.Rectangle([-20.0, -35.0, 55.0, 40.0]),
    'Asia': ee.Geometry.Rectangle([55.0, -10.0, 150.0, 60.0]),
    'Latin America': ee.Geometry.Rectangle([-95.0, -55.0, -30.0, 20.0])
}

def get_region_from_roi(roi):
    centroid = roi.centroid()
    for name, geom in regions.items():
        if geom.contains(centroid).getInfo():
            return name
    return None

def load_region_model(region_name):
    filename = region_models[region_name]

    model_path = hf_hub_download(
        repo_id="Masolele/deforestwatch-models",  # Change if your username/repo differs
        filename=filename,
        cache_dir="models"  # Store locally to avoid repeated downloads
    )

    return load_model(model_path, compile=False)

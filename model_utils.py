import os
import ee
from keras.models import load_model
from huggingface_hub import hf_hub_download
from tensorflow.keras.utils import custom_object_scope
import tensorflow as tf
from Unet_RES_Att_models_IV import Attention_UNetFusion3I, Attention_UNetFusion3I_Sentinel

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

def get_custom_objects():
    """Register all custom layers and operations used in the model"""
    def unpack_and_reshape(x):
        """Custom function to handle unstack operation"""
        return tf.unstack(x, axis=-1)
    
    custom_objects = {
        # Handle TensorFlow operations in Lambda layers
        'TFOpLambda': tf.keras.layers.Lambda,
        'unstack': unpack_and_reshape,  # Custom handler for unstack operation
        
        # Register your custom attention layers
        'Attention_UNetFusion3I': Attention_UNetFusion3I,
        'Attention_UNetFusion3I_Sentinel': Attention_UNetFusion3I_Sentinel,
        
        # Add any other custom operations here
        'tf': tf,  # Provides access to all tf operations
    }
    return custom_objects


def load_region_model(region_name):
    filename = region_models[region_name]

    model_path = hf_hub_download(
        repo_id="Masolele/deforestwatch-models",  # Change if your username/repo differs
        filename=filename,
        repo_type="dataset",  # ⚠️ Important! This tells HF it's a dataset, not a model
        cache_dir="models"  # Store locally to avoid repeated downloads
    )

    #return load_model(model_path, compile=False)
        # Load model with custom objects
    with custom_object_scope(get_custom_objects()):
        model = load_model(model_path, compile=False)
    
    return model

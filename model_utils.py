import os
import ee
from keras.models import load_model
from huggingface_hub import hf_hub_download
from tensorflow.keras.utils import custom_object_scope
import tensorflow as tf
from Unet_RES_Att_models_IV import Attention_UNetFusion3I, Attention_UNetFusion3I_Sentinel,gating_signal,repeat_elem,attention_block

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

# Custom layer registration
def get_custom_objects():
    """Register all custom layers and objects used in the model"""
    def safe_unstack(x, num=None, axis=-1):
        return tf.unstack(x, num=num, axis=axis)
    
    custom_objects = {
        # Register TensorFlow operations used in Lambda layers
        'TFOpLambda': tf.keras.layers.Lambda,
        'tf': tf,
        'safe_unstack': safe_unstack,
        
        # Register your custom attention layers
        'Attention_UNetFusion3I': Attention_UNetFusion3I,
        'Attention_UNetFusion3I_Sentinel': Attention_UNetFusion3I_Sentinel,
        
        # Register any other custom components
        'gating_signal': gating_signal,
        'attention_block': attention_block,
        'repeat_elem': repeat_elem,
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

import os
import ee
from keras.models import load_model
from huggingface_hub import hf_hub_download
from tensorflow.keras.utils import custom_object_scope
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils import get_custom_objects
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

# ✅ Step 1: Map known functions
FUNCTION_MAP = {
    'unstack': tf.unstack,
    'squeeze': tf.squeeze,
    # Add others if needed...
}

# ✅ Step 2: Dynamic Lambda loader
def dynamic_lambda(function):
    if function in FUNCTION_MAP:
        return tf.keras.layers.Lambda(FUNCTION_MAP[function])
    else:
        raise ValueError(f"Unsupported Lambda function: {function}")
        
# Custom handler for TFOpLambda during deserialization
class TFOpLambda(tf.keras.layers.Layer):
    def __init__(self, function=None, **kwargs):
        super().__init__(**kwargs)
        self.function_name = function
        if function not in FUNCTION_MAP:
            raise ValueError(f"Unsupported TFOpLambda function: {function}")
        self.fn = FUNCTION_MAP[function]

    def call(self, inputs):
        return self.fn(inputs)

    def get_config(self):
        return {
            'function': self.function_name,
            **super().get_config()
        }

# # Define dummy fallback if the Lambda logic is unavailable
# class TFOpLambda(tf.keras.layers.Layer):
#     def call(self, inputs):
#         return inputs  # fallback if Lambda logic cannot be restored

def load_region_model(region_name):
    filename = region_models[region_name]

    model_path = hf_hub_download(
        repo_id="Masolele/deforestwatch-models",  # Change if your username/repo differs
        filename=filename,
        repo_type="dataset",  # ⚠️ Important! This tells HF it's a dataset, not a model
        cache_dir="models"  # Store locally to avoid repeated downloads
    )
    with custom_object_scope({'TFOpLambda': TFOpLambda}):
        model = load_model(model_path, compile=False)
    return model


    # # ✅ Register the missing 'TFOpLambda' layer name
    # with custom_object_scope({
    #     'TFOpLambda': TFOpLambda
    # }):
    #     model = load_model(model_path, compile=False)

    # return model

    #return load_model(model_path, compile=False)

import os
import ee
from keras.models import load_model
from huggingface_hub import hf_hub_download
from tensorflow.keras.utils import custom_object_scope
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from keras.utils import get_custom_objects
from Unet_RES_Att_models_IV import Attention_UNetFusion3I, Attention_UNetFusion3I_Sentinel
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K

# Region-to-model file names
region_models = {
    'Africa': 'best_weights_att_unet_lagtime_5_Fused3_2023_totalLoss6V1_without_loss_sentAfrica6.hdf5', #best_weights_VIT_FusionAFR2.keras
    'Asia': 'best_weights_VIT_FusionSEA1.keras',
    'Latin America': 'best_weights_VIT_FusionSA7.keras'
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

custom_objectslamda = {
    "TFOpLambda": Lambda,
}
def unstack_channels(x, num=15, axis=-1):
    return tf.unstack(x, num=num, axis=axis)

def stack_channels1(channels):
    return tf.stack([channels[0], channels[1], channels[2], channels[3], 
                   channels[4], channels[5], channels[6], channels[7],
                   channels[8], channels[9],, channels[10], channels[11]], axis=-1)

def stack_channels2(channels):
    return tf.stack([channels[12], channels[13]], axis=-1)

def stack_channels3(channels):
    return tf.stack([channels[14], channels[15], channels[16]], axis=-1)

custom_objects = {
    'unstack_channels': unstack_channels,
    'stack_channels1': stack_channels1,
    'stack_channels2': stack_channels2,
    'stack_channels3': stack_channels3,
    # Include any other custom functions used in your model
    'gating_signal': gating_signal,
    'attention_block': attention_block
}

@register_keras_serializable()
class PositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.position_embedding = self.add_weight(
            shape=(1, num_patches, embed_dim),
            initializer='random_normal',
            trainable=True,
            name="pos_embedding"
        )

    def call(self, x):
        return x + self.position_embedding

def load_region_model(region_name):
    filename = region_models[region_name]

    model_path = hf_hub_download(
        repo_id="Masolele/deforestwatch-models",  # Change if your username/repo differs
        filename=filename,
        repo_type="dataset",  # ⚠️ Important! This tells HF it's a dataset, not a model
        cache_dir="models"  # Store locally to avoid repeated downloads
    )
    #return load_model(model_path, compile=False)
    return load_model(model_path, compile=False, custom_objects=custom_objects.update({
    'Attention_UNetFusion3I_Sentinel': Attention_UNetFusion3I_Sentinel,custom_objects}))
    #return load_model(model_path, compile=False, custom_objects={'PositionEmbedding': PositionEmbedding})
    

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

# Custom layer registration
def get_custom_objects():
    """Register all custom layers and objects used in the model"""
    def safe_unstack(x, num=None, axis=-1):
        return tf.unstack(x, num=num, axis=axis)
    
    def safe_stack(tensors, axis=-1):
        return tf.stack(tensors, axis=axis)
    
    custom_objects = {
        # Register TensorFlow operations used in Lambda layers
        'TFOpLambda': tf.keras.layers.Lambda,
        'tf': tf,
        'safe_unstack': safe_unstack,
        'safe_stack': safe_stack,
        
        # Register your custom attention layers
        'Attention_UNetFusion3I': Attention_UNetFusion3I,
        'Attention_UNetFusion3I_Sentinel': Attention_UNetFusion3I_Sentinel,
        
        # Register any other custom components
        'gating_signal': gating_signal,
        'attention_block': attention_block,
        'repeat_elem': repeat_elem,
    }
    return custom_objects
    
def get_region_from_roi(roi):
    centroid = roi.centroid()
    for name, geom in regions.items():
        if geom.contains(centroid).getInfo():
            return name
    return None

# # Define dummy fallback if the Lambda logic is unavailable
# class TFOpLambda(tf.keras.layers.Layer):
#     def call(self, inputs):
#         return inputs  # fallback if Lambda logic cannot be restored

def load_region_model(region_name):
    from Unet_RES_Att_models_IV import Attention_UNetFusion3I, Attention_UNetFusion3I_Sentinel
    filename = region_models[region_name]

    model_path = hf_hub_download(
        repo_id="Masolele/deforestwatch-models",  # Change if your username/repo differs
        filename=filename,
        repo_type="dataset",  # ⚠️ Important! This tells HF it's a dataset, not a model
        cache_dir="models"  # Store locally to avoid repeated downloads
    )
    #return load_model(model_path, compile=False)
    # Load model with all custom objects
    with custom_object_scope(get_custom_objects()):
        try:
            # First try loading normally
            model = load_model(model_path, compile=False)
        except (TypeError, ValueError) as e:
            if "unsupported callable" in str(e) or "Unknown layer" in str(e):
                # If that fails, try with safe_mode=False
                model = load_model(model_path, compile=False, safe_mode=False)
            else:
                raise
    
    return model

# Helper functions
def repeat_elem(tensor, rep):
    return layers.Lambda(
        lambda x, repnum: tf.repeat(x, repnum, axis=3),
        arguments={'repnum': rep}
    )(tensor)

def gating_signal(input, out_size, batch_norm=False):
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = tf.shape(x)
    shape_g = tf.shape(gating)
    
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    shape_theta_x = tf.shape(theta_x)
    
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(
        inter_shape, (3, 3),
        strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
        padding='same'
    )(phi_g)
    
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = tf.shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(
        size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2])
    )(sigmoid_xg)
    
    upsample_psi = repeat_elem(upsample_psi, shape_x[3])
    y = layers.multiply([upsample_psi, x])
    
    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

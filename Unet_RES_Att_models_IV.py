
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from keras.layers import Concatenate, Input, Lambda
from tensorflow.keras import backend as K
import numpy as np




'''
A few useful metrics and losses
'''

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


##############################################################
'''
Useful blocks to build Unet

conv - BN - Activation - conv - BN - Activation - Dropout (if enabled)

'''


def conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv

def conv_blockLL(x, filter_size, size, dropout, batch_norm=False):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation("relu")(conv)
    conv = tf.math.sin(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation("relu")(conv)
    conv = tf.math.sin(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv

def conv_blockD(x, filter_size, size, dropout, batch_norm=False, dilation_rate=1):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same", dilation_rate=dilation_rate)(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same", dilation_rate=dilation_rate)(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv
    

def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).
    
    1. conv - BN - Activation - conv - BN - Activation 
                                          - shortcut  - BN - shortcut+BN
                                          
    2. conv - BN - Activation - conv - BN   
                                     - shortcut  - BN - shortcut+BN - Activation                                     
    
    Check fig 4 in https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
    '''

    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path

def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

#multilayer perceptron (MLP)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

#Implementing patch creation as a layer

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size


    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


#Implementing the patch encoding layer
#The PatchEncoder layer will linearly transform a patch by projecting it into a vector of size projection_dim. 
#In addition, it adds a learnable position embedding to the projected vector.
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )


    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def locational_encoder(latlon_input_shape):
    """Build a simple locational encoder to process latitude and longitude."""
    inputs = layers.Input(shape=latlon_input_shape)
    
    # Process latitude and longitude with a few convolutional layers
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Flatten the output and feed it into a dense layer
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    locational_output = layers.Reshape((64, 64, 64))(x)  # Reshape to match U-Net output dimensions
    
    locational_model = models.Model(inputs=inputs, outputs=locational_output, name="Locational_Encoder")
    return locational_model

#Build the ViT Model
# def create_vit_classifier(inputs):
#     #inputs = keras.Input(shape=input_shape)
#     # Augment data.
#     #augmented = data_augmentation(inputs)
#     # Create patches.
#     patches = Patches(patch_size)(inputs) #(augmented)
#     # Encode patches.
#     encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

#     # Create multiple layers of the Transformer block.
#     for _ in range(transformer_layers):
#         # Layer normalization 1.
#         x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#         # Create a multi-head attention layer.
#         attention_output = layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=projection_dim, dropout=0.1
#         )(x1, x1)
#         # Skip connection 1.
#         x2 = layers.Add()([attention_output, encoded_patches])
#         # Layer normalization 2.
#         x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
#         # MLP.
#         x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
#         # Skip connection 2.
#         encoded_patches = layers.Add()([x3, x2])

#     # Create a [batch_size, projection_dim] tensor.
#     representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#     representation = layers.Flatten()(representation)
#     representation = layers.Dropout(0.5)(representation)
#     # Add MLP.
#     features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

#     return features
    
    # # Classify outputs.
    # logits = layers.Dense(num_classes)(features)
    # # Create the Keras model.
    # model = keras.Model(inputs=inputs, outputs=logits)
    # return model



def fourier_transform_layer(x):
    # Forward Fourier Transform
    x_fft = tf.signal.fft2d(tf.cast(x, tf.complex64))
    return x_fft

def inverse_fourier_transform_layer(x):
    # Inverse Fourier Transform
    x_ifft = tf.signal.ifft2d(x)
    return tf.math.abs(x_ifft)

def low_pass_filter(freq, cutoff):
    # Create a low-pass filter by zeroing out high frequencies
    rows, cols = freq.shape[-2], freq.shape[-1]
    mask = np.zeros((rows, cols))
    
    # Define a circular low-pass mask (you can also use other shapes)
    center_row, center_col = rows // 2, cols // 2
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2) < cutoff:
                mask[i, j] = 1
    mask = tf.convert_to_tensor(mask, dtype=tf.complex64)
    return freq * mask

def high_pass_filter(freq, cutoff):
    # Create a high-pass filter by zeroing out low frequencies
    rows, cols = freq.shape[-2], freq.shape[-1]
    mask = np.ones((rows, cols))
    
    # Define a circular high-pass mask (you can also use other shapes)
    center_row, center_col = rows // 2, cols // 2
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2) < cutoff:
                mask[i, j] = 0
    mask = tf.convert_to_tensor(mask, dtype=tf.complex64)
    return freq * mask



def apply_low_pass_filter(x, cutoff):
    x_fft = Lambda(fourier_transform_layer)(x)
    x_low_pass = Lambda(lambda freq: low_pass_filter(freq, cutoff))(x_fft)
    x_filtered = Lambda(inverse_fourier_transform_layer)(x_low_pass)
    return x_filtered

def apply_high_pass_filter(x, cutoff):
    x_fft = Lambda(fourier_transform_layer)(x)
    x_high_pass = Lambda(lambda freq: high_pass_filter(freq, cutoff))(x_fft)
    x_filtered = Lambda(inverse_fourier_transform_layer)(x_high_pass)
    return x_filtered


def unet_block_with_fft_and_filters(inputs):
    #inputs = tf.keras.Input(shape=input_shape)
    
    # Apply low-pass filtering
    low_pass_filtered = apply_low_pass_filter(inputs, cutoff=70)
    
    # Apply high-pass filtering
    high_pass_filtered = apply_high_pass_filter(inputs, cutoff=70)
    
    # Concatenate or combine the filtered outputs
    combined_filters = tf.keras.layers.Concatenate()([low_pass_filtered, high_pass_filtered])
    
    # Followed by the U-Net architecture (example of a simple U-Net)
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(combined_filters)
    # conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    
    # # Additional layers for the U-Net can be added here
    # # Follow the U-Net architecture by adding more convolutional layers, pooling, and upsampling
    
    # # Final layer with softmax for multi-class segmentation
    # outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv1)
    
    # model = tf.keras.Model(inputs, outputs)
    
    return combined_filters #model





# # Example U-Net block (encoder + FFT processing with low-pass and high-pass filters)
# def unet_block_with_fft_and_filters(input_tensor):
#     # Encoder part (convolution + pooling)
#     x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
#     #x = tf.keras.layers.MaxPooling2D((2, 2))(x)

#     # Apply Discrete Fourier Transform on the feature map
#     x_fft = tf.signal.fft2d(tf.cast(x, dtype=tf.complex64))  # Apply 2D FFT

#     # Get the shape of the feature map
#     feature_shape = tf.shape(x)[1:3]  # (height, width)

#     # Create low-pass and high-pass filters for frequency domain processing
#     low_pass_filter, high_pass_filter = create_frequency_filter_tf(feature_shape)

#     # Apply low-pass filter (keep low frequencies)
#     low_passed_fft = x_fft * low_pass_filter

#     # Apply high-pass filter (keep high frequencies)
#     high_passed_fft = x_fft * high_pass_filter

#     # Apply inverse DFT to both low-pass and high-pass filtered results
#     x_low_pass = tf.signal.ifft2d(low_passed_fft)
#     x_high_pass = tf.signal.ifft2d(high_passed_fft)

#     # Take only the real part after inverse FFT (to return to the spatial domain)
#     x_low_pass_real = tf.math.real(x_low_pass)
#     x_high_pass_real = tf.math.real(x_high_pass)

#     # Combine both filtered results by adding them
#     x_filtered = tf.keras.layers.Add()([x_low_pass_real, x_high_pass_real])

#     # Continue with decoder part (up-sampling)
#     x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x_filtered)
#     #x = tf.keras.layers.UpSampling2D((2, 2))(x)
    
#     return x


# Sample U-Net block (encoder + FFT processing + decoder)
def unet_block_with_fft(input_tensor):
    # Encoder part (convolution + pooling)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    # x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Apply Discrete Fourier Transform on the feature map
    x_fft = tf.signal.fft2d(tf.cast(x, dtype=tf.complex64))  # Apply FFT
    
    # Process real and imaginary parts (or magnitude and phase)
    x_real = tf.math.real(x_fft)
    x_imag = tf.math.imag(x_fft)
    
    # Optional: Apply some frequency domain filtering, or process x_real/x_imag further

    # Combine real and imaginary parts again (e.g., concatenate)
    x_fft_processed = tf.keras.layers.Concatenate()([x_real, x_imag])

    # Continue with decoder part (up-sampling)
    # x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x_fft_processed)
    # x = tf.keras.layers.UpSampling2D((2, 2))(x)
    
    return x_fft_processed


# Fourier Transform layer to compute the power spectrum
def compute_power_spectrum(inputs):
    # Compute the 2D Fourier Transform for each band
    fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
    
    # Compute the power spectrum: |fft|^2
    power_spectrum = tf.math.real(fft)**2 + tf.math.imag(fft)**2
    return power_spectrum


# # Transformer encoder block option1
# def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
#     attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
#     attention = layers.Dropout(dropout)(attention)
#     attention = layers.LayerNormalization(epsilon=1e-6)(attention + inputs)

#     ff = layers.Dense(ff_dim, activation="relu")(attention)
#     ff = layers.Dense(inputs.shape[-1])(ff)
#     ff = layers.Dropout(dropout)(ff)
#     ff = layers.LayerNormalization(epsilon=1e-6)(ff + attention)
#     return ff

# # Function to add Transformer layer after U-Net concatenation
# def add_transformer_block(inputs, num_heads=8, ff_dim=2048):
#     # Reshape to (batch_size, sequence_length, embedding_dim)
#     seq_len = inputs.shape[1] * inputs.shape[2]
#     embedding_dim = inputs.shape[3]
    
#     x = layers.Reshape((seq_len, embedding_dim))(inputs)
    
#     # Apply Transformer Encoder
#     x = transformer_encoder(x, num_heads=num_heads, ff_dim=ff_dim)
    
#     # Reshape back to original image-like shape
#     x = layers.Reshape((inputs.shape[1], inputs.shape[2], embedding_dim))(x)
    
#     return x

# Transformer encoder block option2
# Transformer encoder block with positional encoding
def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
    # Positional encoding for spatial information
    pos_encoding = tf.expand_dims(tf.range(start=0, limit=inputs.shape[1], delta=1, dtype=tf.float32), axis=-1)
    inputs += pos_encoding

    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention = layers.Dropout(dropout)(attention)
    attention = layers.LayerNormalization(epsilon=1e-6)(attention + inputs)

    ff = layers.Dense(ff_dim, activation="relu")(attention)
    ff = layers.Dense(inputs.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.LayerNormalization(epsilon=1e-6)(ff + attention)
    return ff

def add_transformer_block(inputs, num_heads=8, ff_dim=2048, dropout_rate=0.1):
    # Reshape to (batch_size, sequence_length, embedding_dim)
    seq_len = inputs.shape[1] * inputs.shape[2]
    embedding_dim = inputs.shape[3]

    x = layers.Reshape((seq_len, embedding_dim))(inputs)
    
    # Apply Transformer Encoder
    x = transformer_encoder(x, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate)
    
    # Reshape back to original image-like shape
    x = layers.Reshape((inputs.shape[1], inputs.shape[2], embedding_dim))(x)
    
    return x


def UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    

    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
   
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
   
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
   
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model 
    model = models.Model(inputs, conv_final, name="UNet")
    #print(model.summary())
    return model

def Attention_UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    
    #channels = tf.unstack (inputs, num=15, axis=-1)
    #inputs  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="Attention_UNet")
    return model

def Attention_UNetFusion(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, axis=-1)
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    #data2
    #inputs2 = layers.Input(input_shape2, dtype=tf.float32)
    #inputs2 = Lambda(lambda x: x[:,:,:, 4:6])(inputs)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    #Concatenate data 1 and 2
    #summed = layers.add([up_conv_128, up2_conv_128])
    merge_data = layers.concatenate([up_conv_128, up2_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Define the U-Net model
def Attention_UNetFusion3I_SentinelTreeHeight(input_shape, dropout_rate=0.0):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    bn = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    bn = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(bn)

    # Decoder
    u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn)
    u3 = layers.concatenate([u3, c3])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c6)

    # Output layer for tree height prediction
    outputs = layers.Conv2D(1, (1, 1), activation='linear', name='tree_height')(c6)

    model = models.Model(inputs = inputs, outputs = inputs, name="UNet_TreeHeight")
    return model



def Attention_UNetFusion3I(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=9, axis=-1)
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    # inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    # inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)
    
    #with forest loss
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[9]], axis=-1)
    # inputs2  = tf.stack ([channels[5], channels[6], channels[9]], axis=-1)
    # inputs3  = tf.stack ([channels[7], channels[8], channels[9]], axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4]], axis=-1)
    inputs2  = tf.stack ([channels[5], channels[6]], axis=-1)
    inputs3  = tf.stack ([channels[7], channels[8]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model


def Attention_UNetFusion3I_Sentinel(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=15, axis=-1)
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    # inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    # inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)
    
    #with forest loss
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[9]], axis=-1)
    # inputs2  = tf.stack ([channels[5], channels[6], channels[9]], axis=-1)
    # inputs3  = tf.stack ([channels[7], channels[8], channels[9]], axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9]], axis=-1)
    #inputs1 =  tf.math.l2_normalize(inputs1n, axis=1, name='norm')
    
    inputs2  = tf.stack ([channels[10], channels[11]], axis=-1)
    #inputs2 =  tf.math.l2_normalize(inputs2n, axis=1, epsilon=1e-12, name='normSAR')
    
    inputs3  = tf.stack ([channels[12], channels[13], channels[14]], axis=-1)
    #inputs3  = tf.stack ([channels[13], channels[14]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model

#######################################################################################################################

def Attention_UNetFusion3I_Sentinel2_Binary(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=17, axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9], channels[10], channels[11]], axis=-1)
    #inputs1 =  tf.math.l2_normalize(inputs1n, axis=1, name='norm')
    
    inputs2  = tf.stack ([channels[12], channels[13]], axis=-1)
    #inputs2 =  tf.math.l2_normalize(inputs2n, axis=1, epsilon=1e-12, name='normSAR')
    
    inputs3  = tf.stack ([channels[14], channels[15], channels[16]], axis=-1)
    #inputs3  = tf.stack ([channels[13], channels[14]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model

##########################################################################################################################
def Attention_UNetFusion3I_Sentinel2(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=17, axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9], channels[10], channels[11]], axis=-1)
    #inputs1 =  tf.math.l2_normalize(inputs1n, axis=1, name='norm')
    
    inputs2  = tf.stack ([channels[12], channels[13]], axis=-1)
    #inputs2 =  tf.math.l2_normalize(inputs2n, axis=1, epsilon=1e-12, name='normSAR')
    
    inputs3  = tf.stack ([channels[14], channels[15], channels[16]], axis=-1)
    #inputs3  = tf.stack ([channels[13], channels[14]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model

##################################################################################################################################################

def RES_Attention_UNetFusion3I_Sentinel(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=15, axis=-1)
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    # inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    # inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)
    
    #with forest loss
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[9]], axis=-1)
    # inputs2  = tf.stack ([channels[5], channels[6], channels[9]], axis=-1)
    # inputs3  = tf.stack ([channels[7], channels[8], channels[9]], axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9]], axis=-1)
    #inputs1 =  tf.math.l2_normalize(inputs1n, axis=1, name='norm')
    
    inputs2  = tf.stack ([channels[10], channels[11]], axis=-1)
    #inputs2 =  tf.math.l2_normalize(inputs2n, axis=1, epsilon=1e-12, name='normSAR')
    
    inputs3  = tf.stack ([channels[12], channels[13], channels[14]], axis=-1)
    #inputs3  = tf.stack ([channels[13], channels[14]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = res_conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = res_conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = res_conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = res_conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = res_conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = res_conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = res_conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = res_conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = res_conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = res_conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = res_conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = res_conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = res_conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = res_conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = res_conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="res_Attention_UNet_Fusion")
    return model


##################
def Attention_UNetFusion3I_SentinelLatLon(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=15, axis=-1)
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    # inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    # inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)
    
    #with forest loss
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[9]], axis=-1)
    # inputs2  = tf.stack ([channels[5], channels[6], channels[9]], axis=-1)
    # inputs3  = tf.stack ([channels[7], channels[8], channels[9]], axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9]], axis=-1)
    #inputs1 =  tf.math.l2_normalize(inputs1n, axis=1, name='norm')
    
    inputs2  = tf.stack ([channels[10], channels[11]], axis=-1)
    #inputs2 =  tf.math.l2_normalize(inputs2n, axis=1, epsilon=1e-12, name='normSAR')
    
    inputs3  = tf.stack ([channels[12], channels[13], channels[14]], axis=-1)
    #inputs3  = tf.stack ([channels[13], channels[14]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)

    # Multi-task outputs  
    # 1*1 convolutional layers
    land_use_output = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1), activation='softmax', name='land_use')(merge_data)
    location_output = layers.Conv2D(2,  kernel_size=(1,1), activation='sigmoid', name='location')(merge_data) # Predicting lat/lon normalized
    #conv_final = layers.BatchNormalization(axis=3)(conv_final)
    #conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = [land_use_output, location_output], name="Attention_UNet_FusionLatLon")
    return model


#################


def Attention_UNetFusion3I_SentinelMLP(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=15, axis=-1)
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    # inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    # inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)
    
    #with forest loss
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[9]], axis=-1)
    # inputs2  = tf.stack ([channels[5], channels[6], channels[9]], axis=-1)
    # inputs3  = tf.stack ([channels[7], channels[8], channels[9]], axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9]], axis=-1)
    #inputs1 =  tf.math.l2_normalize(inputs1n, axis=1, name='norm')
    
    inputs2  = tf.stack ([channels[10], channels[11]], axis=-1)
    #inputs2 =  tf.math.l2_normalize(inputs2n, axis=1, epsilon=1e-12, name='normSAR')
    
    inputs3  = tf.stack ([channels[12], channels[13], channels[14]], axis=-1)
    #inputs3  = tf.stack ([channels[13], channels[14]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    #conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
     # Block 4 (MLP block)
    #mlp_input = layers.Input(shape=(64, 64, 3), name='MLP_input')
    #mlp_input = layers.Input(shape=(64, 64, 3))  # MLP input shape (64, 64, 3)
    x = layers.Flatten()(inputs3)
    x = layers.Dense(32, activation='relu')(x)
    #x = layers.Dropout(0.2)(x)
    x = layers.Dense(64 * 64 * 64, activation='relu')(x)
    mlp_output = layers.Reshape((64, 64, 64))(x)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([6*up_conv_128, 2*up2_conv_128, 1*mlp_output], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model

def Attention_UNetFusion3I_SentinelW(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=15, axis=-1)
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    # inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    # inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)
    
    #with forest loss
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[9]], axis=-1)
    # inputs2  = tf.stack ([channels[5], channels[6], channels[9]], axis=-1)
    # inputs3  = tf.stack ([channels[7], channels[8], channels[9]], axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9]], axis=-1)
    #inputs1 =  tf.math.l2_normalize(inputs1n, axis=1, name='norm')
    
    inputs2  = tf.stack ([channels[10], channels[11]], axis=-1)
    #inputs2 =  tf.math.l2_normalize(inputs2n, axis=1, epsilon=1e-12, name='normSAR')
    
    inputs3  = tf.stack ([channels[12], channels[13], channels[14]], axis=-1)
    #inputs3  = tf.stack ([channels[13], channels[14]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    #merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)

    # Initialize weights as trainable variables
    #weight1 = tf.Variable(1.0, trainable=True)
    #weight2 = tf.Variable(1.0, trainable=True)
    #weight3 = tf.Variable(1.0, trainable=True)

    # or Define weight for the first block
    weight1 = 1.2  # Give more weight to the first input/block
    weight2 = 1.0
    weight3 = 1.5
    
    # Multiply outputs by their respective weights
    weighted_output1 = weight1 * up_conv_128
    weighted_output2 = weight2 * up2_conv_128
    weighted_output3 = weight3 * up3_conv_128
    
    # Apply the weights to the outputs
    #weighted_sum = weight1 * output1 + weight2 * output2 + weight3 * output3
    combined_output = layers.concatenate([weighted_output1, weighted_output2, weighted_output3], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(combined_output)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model

  
###########################dilated conv##################
def DilatedAttention_UNetFusion3I_Sentinel(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet with multi-branch and dilation for larger context awareness
    '''
    # network structure
    FILTER_NUM = 64
    FILTER_NUM3 = 16
    FILTER_SIZE = 3
    UP_SAMP_SIZE = 2
    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    channels = tf.unstack(inputs, num=15, axis=-1)
    
    inputs1 = tf.stack([channels[i] for i in range(10)], axis=-1)
    inputs2 = tf.stack([channels[10], channels[11]], axis=-1)
    inputs3 = tf.stack([channels[12], channels[13], channels[14]], axis=-1)

    ### Downsampling Path with Dilations ###
    # Branch 1 (input1)
    conv_128 = conv_blockD(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    
    conv_32 = conv_blockD(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm, dilation_rate=2)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)

    conv_16 = conv_blockD(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm, dilation_rate=4)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    
    conv_8 = conv_blockD(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm, dilation_rate=8)
    
    # Branch 2 (input2)
    conv2_128 = conv_blockD(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    
    conv2_32 = conv_blockD(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm, dilation_rate=2)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)

    conv2_16 = conv_blockD(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm, dilation_rate=4)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    
    conv2_8 = conv_blockD(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm, dilation_rate=8)
    

    ### Upsampling Path ###

    # Upsampling layers1
    # UpRes , attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
    # Upsampling layers 2
    # UpRes , attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 

    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ### Global Context Branch ###
    global_pool = layers.GlobalAveragePooling2D()(conv_8)
    global_features = layers.Dense(64, activation='relu')(global_pool)
    global_features = layers.Reshape((1, 1, 64))(global_features)
    global_features = layers.UpSampling2D(size=(64, 64))(global_features)
    
    ### Merge branches and global features ###
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, global_features], axis=-1)

    # Process latitude and longitude with a few convolutional layers
    locatn = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs3)
    locatn = layers.BatchNormalization()(locatn)
    locatn = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(locatn)
    locatn = layers.BatchNormalization()(locatn)
    locatn = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(locatn)
    locational_features = layers.BatchNormalization()(locatn)
        # Global average pooling to reduce dimensionality
    #locational_features = layers.GlobalAveragePooling2D()(locatn)
    
    # Flatten the output and feed it into a dense layer
    # locatn = layers.Flatten()(locatn)
    # locatn = layers.Dense(512, activation='relu')(locatn)
    # locational_features = layers.Reshape((64, 64, 128))(locatn)  # Reshape to match U-Net output dimensions

    # Concatenate locational features with U-Net output
    merge_with_location = layers.concatenate([merge_data, locational_features], axis=-1)

    # Final convolution and output
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_with_location)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  # For multiclass segmentation

    # Model integration
    model = models.Model(inputs=inputs, outputs=conv_final, name="Attention_UNet_Fusion_Large_Context")
    
    return model


###########################Transformer###########################################

def TransformerAttention_UNetFusion3I_Sentinel(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=15, axis=-1)
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    # inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    # inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)
    
    #with forest loss
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[9]], axis=-1)
    # inputs2  = tf.stack ([channels[5], channels[6], channels[9]], axis=-1)
    # inputs3  = tf.stack ([channels[7], channels[8], channels[9]], axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9]], axis=-1)
    #inputs1 =  tf.math.l2_normalize(inputs1n, axis=1, name='norm')
    
    inputs2  = tf.stack ([channels[10], channels[11]], axis=-1)
    #inputs2 =  tf.math.l2_normalize(inputs2n, axis=1, epsilon=1e-12, name='normSAR')
    
    inputs3  = tf.stack ([channels[12], channels[13], channels[14]], axis=-1)
    #inputs3  = tf.stack ([channels[13], channels[14]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)

    # Add Transformer block #option 2
    #transformer_output = add_transformer_block(merge_data)
    #Adding transformer bock with optimisation #option2
    transformer_output = add_transformer_block(merge_data, num_heads=16, ff_dim=4096, dropout_rate=0.2)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(transformer_output)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion_transformer")
    return model

###########################Descrete Fourier transform based Attention unet##########################

def DFTAttention_UNetFusion3I_Sentinel(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=15, axis=-1)
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    # inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    # inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)
    
    #with forest loss
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[9]], axis=-1)
    # inputs2  = tf.stack ([channels[5], channels[6], channels[9]], axis=-1)
    # inputs3  = tf.stack ([channels[7], channels[8], channels[9]], axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9]], axis=-1)
    #inputs1 =  tf.math.l2_normalize(inputs1n, axis=1, name='norm')
    
    inputs2  = tf.stack ([channels[10], channels[11]], axis=-1)
    #inputs2 =  tf.math.l2_normalize(inputs2n, axis=1, epsilon=1e-12, name='normSAR')
    
    inputs3  = tf.stack ([channels[12], channels[13], channels[14]], axis=-1)
    #inputs3  = tf.stack ([channels[13], channels[14]], axis=-1)

    #DFT = unet_block_with_fft_and_filters(inputs1) 
        # Step 1: Compute power spectrum for each band
    DFT = compute_power_spectrum(inputs1)
    #Concatenate block 1, 2 and 3
    inputs1DFT = layers.concatenate([inputs1, DFT], axis=-1)
    
    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1DFT, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################ DFT BLOCK##############################################################
        # Downsampling layers
    # DownRes 1, convolution + pooling
    #conv_128DFT = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    #pool_64DFT = layers.MaxPooling2D(pool_size=(2,2))(conv_128DFT)

    ##DFT

    #DFT = unet_block_with_fft_and_filters(pool_64DFT) 
    #DFT = unet_block_with_fft(pool_64DFT)
    #DFT = unet_block_with_dwt(pool_64DFT)
    
    # # DownRes 3
    # conv_32DFT = conv_block(DFT, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_16DFT = layers.MaxPooling2D(pool_size=(2,2))(conv_32DFT)
    # # DownRes 4
    # conv_16DFT = conv_block(pool_16DFT, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # pool_8DFT = layers.MaxPooling2D(pool_size=(2,2))(conv_16DFT)
    # # DownRes 5, convolution only
    # conv_8DFT = conv_block(pool_8DFT, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # # Upsampling layers
    # # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    # gating_16DFT = gating_signal(conv_8DFT, 4*FILTER_NUM, batch_norm)
    # att_16DFT = attention_block(conv_16DFT, gating_16DFT, 4*FILTER_NUM)
    # up_16DFT = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8DFT)
    # up_16DFT = layers.concatenate([up_16DFT, att_16DFT], axis=3)
    # up_conv_16DFT = conv_block(up_16DFT, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # # UpRes 7
    # gating_32DFT = gating_signal(up_conv_16DFT, 2*FILTER_NUM, batch_norm)
    # att_32DFT = attention_block(conv_32DFT, gating_32DFT, 2*FILTER_NUM)
    # up_32DFT = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16DFT)
    # up_32DFT = layers.concatenate([up_32DFT, att_32DFT], axis=3)
    # up_conv_32DFT = conv_block(up_32DFT, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # # UpRes 9
    # gating_128DFT = gating_signal(up_conv_32DFT, FILTER_NUM, batch_norm)
    # att_128DFT = attention_block(conv_128DFT, gating_128DFT, FILTER_NUM)
    # up_128DFT = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32DFT)
    # up_128DFT = layers.concatenate([up_128DFT, att_128DFT], axis=3)
    # up_conv_128DFT = conv_block(up_128DFT, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
    ############################MODEL BLOCK 2#########################################

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model




def Attention_UNetFusion2I_Sentinel(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=13, axis=-1)
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    # inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    # inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)
    
    #with forest loss
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[9]], axis=-1)
    # inputs2  = tf.stack ([channels[5], channels[6], channels[9]], axis=-1)
    # inputs3  = tf.stack ([channels[7], channels[8], channels[9]], axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[5], channels[6], channels[7], channels[8], channels[9]], axis=-1)
    #inputs1 =  tf.math.l2_normalize(inputs1n, axis=1, name='norm')
    
    #inputs2  = tf.stack ([channels[10], channels[11]], axis=-1)
    #inputs2 =  tf.math.l2_normalize(inputs2n, axis=1, epsilon=1e-12, name='normSAR')
    
    inputs3  = tf.stack ([channels[10], channels[11], channels[12]], axis=-1)
    #inputs3  = tf.stack ([channels[13], channels[14]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ##MODEL BLOCK 2

    # # Downsampling layers
    # # DownRes 1, convolution + pooling
    # conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    # pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # # DownRes 2
    # # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # # DownRes 3
    # conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # # DownRes 4
    # conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # # DownRes 5, convolution only
    # conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # # Upsampling layers
    # # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    # gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    # att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    # up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    # up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    # up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # # UpRes 7
    # gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    # att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    # up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    # up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    # up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # # UpRes 8
    # # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # # UpRes 9
    # gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    # att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    # up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    # up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    # up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model





def Attention_UNetFusion3I_MLP(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 8 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    image_size = 16
    patch_size = 8
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 32
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim,] # size of the transformer layers
    transformer_layers = 8 
    mlp_head_units = [2048, 1024, ] #size of the dense layers of the final classifier

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=9, axis=-1)
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    # inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    # inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)
    
    #with forest loss
    # inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4], channels[9]], axis=-1)
    # inputs2  = tf.stack ([channels[5], channels[6], channels[9]], axis=-1)
    # inputs3  = tf.stack ([channels[7], channels[8], channels[9]], axis=-1)
    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4]], axis=-1)
    inputs2  = tf.stack ([channels[5], channels[6]], axis=-1)
    inputs3  = tf.stack ([channels[7], channels[8]], axis=-1)
    
    # lat_long  = tf.stack ([channels[7], channels[8]], axis=-1)
    # inputs3 = layers.Cropping2D(cropping=((64, 63), (64, 63)))(lat_long)
    
    #inputs3 = layers.Flatten()(lat_long_cropped)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)


    # """ Patch + Position Embeddings """
    # patch_embed = layers.Dense(hidden_dim)(conv_8) ## (None, 256, 768)

    # positions = tf.range(start=0, limit=num_patches, delta=1) ## (256,)
    # pos_embed = layers.Embedding(input_dim=num_patches, output_dim=hidden_dim)(positions) ## (256, 768)
    # PatchPos = patch_embed + pos_embed ## (None, 256, 768)

    # """ Transformer Encoder """
    # x = L.LayerNormalization()(x)
    # x = L.MultiHeadAttention(num_heads=12, key_dim=hidden_dim)(x, x)
    # x = L.LayerNormalization()(x)
    # X = layers.Dense(hidden_dim)(8*FILTER_NUM)
    # z3 = L.Reshape(shape)(z3)

    
    #inputs = keras.Input(shape=input_shape)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    #conv_8V2 = conv_block(conv_8, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    patchesP = Patches(patch_size)(conv_8) #(augmented)
    # Encode patches.
    encoded_patchesP = PatchEncoder(num_patches, projection_dim)(patchesP)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1P = layers.LayerNormalization(epsilon=1e-6)(encoded_patchesP)
        # Create a multi-head attention layer.
        attention_outputP = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1P, x1P)
        # Skip connection 1.
        x2P = layers.Add()([attention_outputP, encoded_patchesP])
        # Layer normalization 2.
        x3P = layers.LayerNormalization(epsilon=1e-6)(x2P)
        # MLP.
        x3P = mlp(x3P, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patchesP = layers.Add()([x3P, x2P])

    # Create a [batch_size, projection_dim] tensor.
    representationP = layers.LayerNormalization(epsilon=1e-6)(encoded_patchesP)
    representationP = layers.Flatten()(representationP)
    representationP = layers.Dropout(0.5)(representationP)
    # Add MLP.
    featuresP = mlp(representationP, hidden_units=mlp_head_units, dropout_rate=0.5)

    #vit_classifier = create_vit_classifier(conv_8)#(conv_8)
    #result = tf.reshape(inputs, (tf.shape(inputs)[0],) #tf.shape(images)[0]

    #ReshapeTP = tf.reshape(featuresP, tf.shape(conv_8))
    ReshapeTP = tf.reshape(featuresP, (-1, image_size, image_size, projection_dim * 2))

    
    #ReshapeT = layers.Reshape(conv_8.shape)(features)
    #ncoded_patches = layers.Add()([x3, x2])
    conv_8P = layers.concatenate([conv_8, ReshapeTP], axis=3)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8P, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8P)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)


    #Transformer
    #conv2_8V2 = conv_block(conv2_8, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    patchesSAR = Patches(patch_size)(conv2_8) #(augmented)
    # Encode patches.
    encoded_patchesSAR = PatchEncoder(num_patches, projection_dim)(patchesSAR)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1SAR = layers.LayerNormalization(epsilon=1e-6)(encoded_patchesSAR)
        # Create a multi-head attention layer.
        attention_outputSAR = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1SAR, x1SAR)
        # Skip connection 1.
        x2SAR = layers.Add()([attention_outputSAR, encoded_patchesSAR])
        # Layer normalization 2.
        x3SAR = layers.LayerNormalization(epsilon=1e-6)(x2SAR)
        # MLP.
        x3SAR = mlp(x3SAR, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patchesSAR = layers.Add()([x3SAR, x2SAR])

    # Create a [batch_size, projection_dim] tensor.
    representationSAR = layers.LayerNormalization(epsilon=1e-6)(encoded_patchesSAR)
    representationSAR = layers.Flatten()(representationSAR)
    representationSAR = layers.Dropout(0.5)(representationSAR)
    # Add MLP.
    featuresSAR = mlp(representationSAR, hidden_units=mlp_head_units, dropout_rate=0.5)

    #vit_classifier = create_vit_classifier(conv_8)#(conv_8)
    #result = tf.reshape(inputs, (tf.shape(inputs)[0],) #tf.shape(images)[0]

    #ReshapeTSAR = tf.reshape(featuresSAR, tf.shape(conv2_8))
    ReshapeTSAR = tf.reshape(featuresSAR, (-1, image_size, image_size, projection_dim * 2))

    #ReshapeT = layers.Reshape(conv_8.shape)(features)
    #ncoded_patches = layers.Add()([x3, x2])
    conv2_8SAR = layers.concatenate([conv2_8, ReshapeTSAR], axis=3)
    

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating2_16 = gating_signal(conv2_8SAR, 4*FILTER_NUM, batch_norm)
    att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8SAR)
    up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers


    #MLP
    # x = layers.Dense(32, activation="gelu")(inputs3)
    # x = layers.Dense(64, activation="gelu")(x)
    # x = layers.Dense(128, activation="gelu")(x)
    # x = layers.Dense(256, activation="gelu")(x)
    # #x = layers.Dropout(0.1)(x)
    # x = layers.Dense(512)(x)
    # up3_conv_128 = layers.UpSampling2D(size=(128, 128), data_format="channels_last")(x)

    
    #Concatenate block 1, 2 and 3
    # merge_data = layers.concatenate([up_conv_128, up2_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model





##Planet only
def Attention_UNetFusion3IPlanet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 #16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, num=7, axis=-1)

    
    #without forest loss
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3], channels[4]], axis=-1)
    #inputs2  = tf.stack ([channels[5], channels[6]], axis=-1)
    inputs3  = tf.stack ([channels[5], channels[6]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    ############################MODEL BLOCK 2
     #MODEL BLOCK 2
    # Downsampling layers
    # DownRes 1, convolution + pooling
    # conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    # pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # # DownRes 2
    # # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # # DownRes 3
    # conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # # DownRes 4
    # conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # # DownRes 5, convolution only
    # conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # # Upsampling layers
    # # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    # gating2_16 = gating_signal(conv2_8, 4*FILTER_NUM, batch_norm)
    # att2_16 = attention_block(conv2_16, gating2_16, 4*FILTER_NUM)
    # up2_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv2_8)
    # up2_16 = layers.concatenate([up2_16, att2_16], axis=3)
    # up2_conv_16 = conv_block(up2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # # UpRes 7
    # gating2_32 = gating_signal(up2_conv_16, 2*FILTER_NUM, batch_norm)
    # att2_32 = attention_block(conv2_32, gating2_32, 2*FILTER_NUM)
    # up2_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_16)
    # up2_32 = layers.concatenate([up2_32, att2_32], axis=3)
    # up2_conv_32 = conv_block(up2_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # # UpRes 8
    # # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # # UpRes 9
    # gating2_128 = gating_signal(up2_conv_32, FILTER_NUM, batch_norm)
    # att2_128 = attention_block(conv2_128, gating2_128, FILTER_NUM)
    # up2_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up2_conv_32)
    # up2_128 = layers.concatenate([up2_128, att2_128], axis=3)
    # up2_conv_128 = conv_block(up2_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating3_16 = gating_signal(conv3_8, 4*FILTER_NUM3, batch_norm)
    att3_16 = attention_block(conv3_16, gating3_16, 4*FILTER_NUM3)
    up3_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(conv3_8)
    up3_16 = layers.concatenate([up3_16, att3_16], axis=3)
    up3_conv_16 = conv_block(up3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 7
    gating3_32 = gating_signal(up3_conv_16, 2*FILTER_NUM3, batch_norm)
    att3_32 = attention_block(conv3_32, gating3_32, 2*FILTER_NUM3)
    up3_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_16)
    up3_32 = layers.concatenate([up3_32, att3_32], axis=3)
    up3_conv_32 = conv_block(up3_32, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating3_128 = gating_signal(up3_conv_32, FILTER_NUM3, batch_norm)
    att3_128 = attention_block(conv3_128, gating3_128, FILTER_NUM3)
    up3_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last", interpolation='bilinear')(up3_conv_32)
    up3_128 = layers.concatenate([up3_128, att3_128], axis=3)
    up3_conv_128 = conv_block(up3_128, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)

    
    #Concatenate block 1, 2 and 3
    merge_data = layers.concatenate([up_conv_128, up3_conv_128], axis=-1)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(merge_data)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_Fusion")
    return model



def Attention_UNetFusionEncoder(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_NUM3 = 16 # number of basic filters for the third model block
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters

    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    #inputs1 = Lambda(lambda x: x[:,:,:, :4])(inputs)
    channels = tf.unstack (inputs, axis=-1)
    inputs1  = tf.stack ([channels[0], channels[1], channels[2], channels[3]], axis=-1)
    inputs2  = tf.stack ([channels[4], channels[5]], axis=-1)
    inputs3  = tf.stack ([channels[6], channels[7]], axis=-1)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    
    ############################MODEL BLOCK 2

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv2_128 = conv_block(inputs2, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool2_64 = layers.MaxPooling2D(pool_size=(2,2))(conv2_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv2_32 = conv_block(pool2_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2_16 = layers.MaxPooling2D(pool_size=(2,2))(conv2_32)
    # DownRes 4
    conv2_16 = conv_block(pool2_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool2_8 = layers.MaxPooling2D(pool_size=(2,2))(conv2_16)
    # DownRes 5, convolution only
    conv2_8 = conv_block(pool2_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)


    
     #MODEL BLOCK 3
    #inputs3 = layers.Input(input_shape3, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv3_128 = conv_block(inputs3, FILTER_SIZE, FILTER_NUM3, dropout_rate, batch_norm)
    pool3_64 = layers.MaxPooling2D(pool_size=(2,2))(conv3_128)
    # DownRes 2
    # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv3_32 = conv_block(pool3_64, FILTER_SIZE, 2*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_16 = layers.MaxPooling2D(pool_size=(2,2))(conv3_32)
    # DownRes 4
    conv3_16 = conv_block(pool3_16, FILTER_SIZE, 4*FILTER_NUM3, dropout_rate, batch_norm)
    pool3_8 = layers.MaxPooling2D(pool_size=(2,2))(conv3_16)
    # DownRes 5, convolution only
    conv3_8 = conv_block(pool3_8, FILTER_SIZE, 8*FILTER_NUM3, dropout_rate, batch_norm)

    #Concatenate Encoder block 1, 2 and 3
    conv_16_merged = layers.concatenate([conv_16, conv2_16, conv3_16], axis=-1)
    conv_32_merged = layers.concatenate([conv_32, conv2_32, conv3_32], axis=-1)
    conv_128_merged = layers.concatenate([conv_128, conv2_128, conv3_128], axis=-1)
    conv_16_merged = layers.concatenate([conv_16, conv2_16, conv3_16], axis=-1)
    
    conv_8_merged = layers.concatenate([conv_8, conv2_8, conv3_8], axis=-1)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8_merged, 4*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16_merged, gating_16, 4*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8_merged)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32_merged, gating_32, 2*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    # up_64 = layers.concatenate([up_64, att_64], axis=3)
    # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128_merged, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs = inputs, outputs = conv_final, name="Attention_UNet_FusionEncoder")
    return model





def Attention_UNetP(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    branch_outputs = []
    for i in range(input_shape[2]):
        
        out = Lambda(lambda x: x[:,:,:, i:i+1])(inputs)
        #Downsampling layers
        # DownRes 1, convolution + pooling
        conv_128 = conv_block(out, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
        pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
        # DownRes 2
        # conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
        # pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
        # DownRes 3
        conv_32 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
        pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
        # DownRes 4
        conv_16 = conv_block(pool_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
        pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
        # DownRes 5, convolution only
        conv_8 = conv_block(pool_8, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

        # Upsampling layers
        # UpRes 6, attention gated concatenation + upsampling + double residual convolution
        gating_16 = gating_signal(conv_8, 4*FILTER_NUM, batch_norm)
        att_16 = attention_block(conv_16, gating_16, 4*FILTER_NUM)
        up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
        up_16 = layers.concatenate([up_16, att_16], axis=3)
        up_conv_16 = conv_block(up_16, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 7
        gating_32 = gating_signal(up_conv_16, 2*FILTER_NUM, batch_norm)
        att_32 = attention_block(conv_32, gating_32, 2*FILTER_NUM)
        up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
        up_32 = layers.concatenate([up_32, att_32], axis=3)
        up_conv_32 = conv_block(up_32, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 8
        # gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
        # att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
        # up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
        # up_64 = layers.concatenate([up_64, att_64], axis=3)
        # up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 9
        gating_128 = gating_signal(up_conv_32, FILTER_NUM, batch_norm)
        att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
        up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
        up_128 = layers.concatenate([up_128, att_128], axis=3)
        up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
        
        branch_outputs.append(up_conv_128)
    
    up_conv_128_2 = Concatenate()(branch_outputs)
    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128_2)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="Attention_UNet")
    return model






def Attention_ResUNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Rsidual UNet, with attention 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    # input data
    # dimension of the image depth
    inputs = layers.Input(input_shape, dtype=tf.float32)
    axis = 3

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = res_conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = res_conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = res_conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = res_conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = res_conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=axis)
    up_conv_16 = res_conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=axis)
    up_conv_32 = res_conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=axis)
    up_conv_64 = res_conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=axis)
    up_conv_128 = res_conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=axis)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="AttentionResUNet")
    return model

input_shape = (256,256,1)
UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True)




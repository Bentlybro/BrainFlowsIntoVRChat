import tensorflow as tf
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Layer, Conv2D, DepthwiseConv2D, SeparableConv2D , Conv1D
from keras.layers import Activation, Multiply, BatchNormalization, SpatialDropout1D, UpSampling2D, GlobalAveragePooling2D, LayerNormalization
from keras.losses import MeanSquaredError as MSE

## Spatial Attention (Thanks Summer!)
@keras.saving.register_keras_serializable()
class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.attn = SeparableConv2D(1, kernel_size, padding='same', activation='sigmoid')
    
    def build(self, input_shape):
        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=-1)
        x = self.attn(x)
        return Multiply()([inputs, x])

## Encoder and Decoder Trained on the physionet motor imagery dataset
## https://www.physionet.org/content/eegmmidb/1.0.0/
## Thanks again to Summer, Programmerboi, Hosomi
## Modification to follow along this paper
## https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-020-00203-9

@keras.saving.register_keras_serializable()
class ExpandDimsLayer(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=-1)

@keras.saving.register_keras_serializable()
class SqueezeDimsLayer(Layer):
    def call(self, inputs):
        return tf.squeeze(inputs, axis=-1)

# Noise Layer 
@keras.utils.register_keras_serializable()
class AddNoiseLayer(Layer):
    def __init__(self, noise_factor=0.1, **kwargs):
        super(AddNoiseLayer, self).__init__(**kwargs)
        self.noise_factor = noise_factor

    def call(self, inputs, training=None):
        if training:
            noise = self.noise_factor * tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=1.0)
            return inputs + noise
        return inputs

kernel = (2, 3)
e_rates = [1, 2]
d_rates = list(reversed(e_rates))
act = 'elu'

@keras.utils.register_keras_serializable()
class StackedDepthSeperableConv2D(Layer):
    def __init__(self, filters, kernel_size, dilation_rates, stride=1, use_residual=False, **kwargs):
        super(StackedDepthSeperableConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.dilation_rates = dilation_rates
        self.depthwise_stack = Sequential([DepthwiseConv2D(kernel_size, padding='same', dilation_rate=dr) for dr in dilation_rates])
        self.pointwise_conv = Conv2D(filters, 1, padding='same', strides=stride)
        self.residual_conv = None
        if use_residual:
            self.residual_conv = Conv2D(filters, 1, padding='same', strides=stride)
    
    def call(self, inputs):
        output = self.depthwise_stack(inputs)
        output = self.pointwise_conv(output)
        if self.residual_conv:
            output += self.residual_conv(inputs)
        return output
    
    def build(self, input_shape):
        super(StackedDepthSeperableConv2D, self).build(input_shape)


encoder = Sequential([
    ExpandDimsLayer(),
    StackedDepthSeperableConv2D(16, kernel, e_rates, 2, True),
    BatchNormalization(), Activation(act), # (80, 32, 16)
    
    StackedDepthSeperableConv2D(16, kernel, e_rates, 2, True),
    BatchNormalization(), Activation(act), # (40, 16, 16)
    
    StackedDepthSeperableConv2D(16, kernel, e_rates, 2, True),
    BatchNormalization(), Activation(act), # (20, 8, 16)

    StackedDepthSeperableConv2D(16, kernel, e_rates, 1, False), 
    Activation('linear')
])

decoder = Sequential([
    StackedDepthSeperableConv2D(16, kernel, d_rates, 1, True),
    BatchNormalization(), Activation(act), UpSampling2D(2),
    
    StackedDepthSeperableConv2D(16, kernel, d_rates, 1, True),
    BatchNormalization(), Activation(act), UpSampling2D(2),

    StackedDepthSeperableConv2D(16, kernel, d_rates, 1, True),
    BatchNormalization(), Activation(act), UpSampling2D(2),
    
    StackedDepthSeperableConv2D(1, kernel, d_rates, 1, False), Activation('linear'),
    SqueezeDimsLayer()
])

## AutoEncoder Wrapper for edf_train
class CustomAutoencoder(Model):
    def __init__(self, encoder, decoder, perceptual_weight=1.0, sd_rate=0.2):
        super(CustomAutoencoder, self).__init__()
        self.spatial_dropout = SpatialDropout1D(sd_rate)
        self.encoder = encoder
        self.decoder = decoder
        self.perceptual_weight = perceptual_weight
        self.mse_loss = MSE()

    def call(self, inputs):
        # Encoding and reconstructing the input
        inputs = self.spatial_dropout(inputs)
        original_features = self.encoder(inputs)
        reconstruction = self.decoder(original_features)
        
        # get features from reconstruction
        reconstructed_features = self.encoder(reconstruction)

        # Compute and add perceptual loss during the call
        perceptual_loss = self.mse_loss(original_features, reconstructed_features)
        self.add_loss(self.perceptual_weight * perceptual_loss)

        # Return only the reconstruction for the main loss computation
        return reconstruction
    
auto_encoder = CustomAutoencoder(encoder, decoder)

## First Layer to convert any channels to 64 ranged [0, 1]
def create_first_layer(chs=64):
    return Sequential([
        AddNoiseLayer(0.2),
        Conv1D(chs, 3, padding='causal', dilation_rate=1),
        LayerNormalization(), Activation(act),
        Conv1D(chs, 3, padding='causal', dilation_rate=2),
        LayerNormalization(), Activation(act), 
        Conv1D(chs, 3, padding='causal', dilation_rate=4),
        Activation('linear'),
    ])

## Last Layer to map latent space to custom classes
def create_last_layer(classes):
    return Sequential([
        GlobalAveragePooling2D(),
        Dense(classes, activation='softmax', kernel_regularizer='l2')
    ])
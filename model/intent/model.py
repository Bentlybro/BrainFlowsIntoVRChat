import tensorflow as tf
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Layer, DepthwiseConv1D, Conv1D, Attention
from keras.layers import Activation, Multiply, BatchNormalization, SpatialDropout1D, UpSampling1D, GlobalAveragePooling1D, Dropout
from keras.losses import MeanSquaredError as MSE

## Spatial Attention (Thanks Summer!)
@keras.utils.register_keras_serializable()
class SpatialAttention(Layer):
    def __init__(self, classes, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.classes = classes
        self.conv1 = Conv1D(self.classes, self.kernel_size, padding='same', activation='elu')
        self.conv2 = Conv1D(1, self.kernel_size, padding='same', activation='sigmoid')
    
    def build(self, input_shape):
        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=2)
        x = self.conv1(x)
        x = self.conv2(x)
        return Multiply()([inputs, x])

@keras.utils.register_keras_serializable()
class SpatialAttentionPool(SpatialAttention):
    def call(self, inputs):
        x = super(SpatialAttentionPool, self).call(inputs)
        return tf.reduce_sum(x, axis=1)

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

## Encoder and Decoder Trained on the physionet motor imagery dataset
## https://www.physionet.org/content/eegmmidb/1.0.0/
## Thanks again to Summer, Programmerboi, Hosomi

kernel = 3
filters = 32
e_rates = [1, 2, 4]
d_rates = list(reversed(e_rates))
act = 'elu'
hidden_layers = 3 # (160, 64) => (20, 32) = 640

## Modification of seperable convolutions to follow along this paper
## https://journalofcloudcomputing.springeropen.com/articles/10.1186/s13677-020-00203-9
@keras.utils.register_keras_serializable()
class StackedDepthSeperableConv1D(Layer):
    def __init__(self, filters, kernel_size, dilation_rates, stride=1, use_residual=False, **kwargs):
        super(StackedDepthSeperableConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.dilation_rates = dilation_rates
        self.depthwise_stack = Sequential([DepthwiseConv1D(kernel_size, padding='same', dilation_rate=dr) for dr in dilation_rates])
        self.pointwise_conv = Conv1D(filters, 1, padding='same', strides=stride)
        self.residual_conv = None
        if use_residual:
            self.residual_conv = Conv1D(filters, 1, padding='same', strides=stride)
    
    def call(self, inputs):
        depthwise_output = self.depthwise_stack(inputs)
        output = self.pointwise_conv(depthwise_output)
        if self.residual_conv:
            output += self.residual_conv(inputs)
        return output
    
    def build(self, input_shape):
        super(StackedDepthSeperableConv1D, self).build(input_shape)

def encoder_blocks(filter, kernel, dilation_rates, strides, use_residual):
    return Sequential([
        StackedDepthSeperableConv1D(filter, kernel, dilation_rates, strides, use_residual),
        BatchNormalization(), Activation(act)
    ])

def decoder_blocks(filter, kernel, dilation_rates, up_rate, use_residual):
    return Sequential([
        StackedDepthSeperableConv1D(filter, kernel, dilation_rates, 1, use_residual),
        BatchNormalization(), Activation(act), UpSampling1D(up_rate)
    ])

encoder = Sequential(
    [encoder_blocks(filters, kernel, e_rates, 2, True) for _ in range(hidden_layers)] + 
    [StackedDepthSeperableConv1D(filters, kernel, e_rates, 1, False), Activation(act)]
)

decoder = Sequential(
    [decoder_blocks(filters, kernel, d_rates, 2, True) for _ in range(hidden_layers)] +
    [StackedDepthSeperableConv1D(64, kernel, d_rates, 1, False), Activation('linear')]
)  

## AutoEncoder Wrapper for edf_train
## Tunes for both feature and reconstruction losses
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

# Custom Activation to maintain zero centered with max standard deviation of 3
def tanh3(x):
    return tf.nn.tanh(x) * 3.5
keras.utils.get_custom_objects().update({'tanh3': tanh3})

def create_classifier(encoder, classes):
    e_input_chans = encoder.input_shape[-1]
    e_output_chans = encoder.output_shape[-1]
    return Sequential([
        # Expansion Block
        Sequential([
            Conv1D(e_input_chans, kernel, padding='causal', dilation_rate=1), Activation(act),
            Conv1D(e_input_chans, kernel, padding='causal', dilation_rate=2), Activation(act), 
            Conv1D(e_input_chans, kernel, padding='causal', dilation_rate=4), Activation(tanh3),
            SpatialDropout1D(0.2)
        ]),

        encoder,

        # Classification Block
        Sequential([
            SpatialAttentionPool(classes),
            Dropout(0.5),
            Dense(e_output_chans, activation=act),
            Dropout(0.5),
            Dense(classes, activation='softmax', kernel_regularizer='l2')
        ])
    ])

import tensorflow as tf
import keras
import numpy as np

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
hidden_layers = 4 # (160, 64) => (10, 32) = 320

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
class CustomAutoencoder(Model):
    def __init__(self, encoder, decoder, max_noise=0.9, min_noise=0.1, rate=1.0, dropout=0.2):
        super(CustomAutoencoder, self).__init__()
        self.max_noise = max_noise
        self.min_noise = min_noise
        self.rate = rate
        self.twopi = tf.constant(2.0 * np.pi)
        self.step = 0.0

        self.dropout = SpatialDropout1D(dropout)
        self.encoder = encoder
        self.decoder = decoder

        self.mse_loss = MSE()

    
    def add_noise(self, inputs, step):
        sin_mod = tf.sin(self.twopi * step)**2
        noise_factor = (self.max_noise - self.min_noise) * sin_mod + self.min_noise
        noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=noise_factor)
        return noise + inputs, noise

    def call(self, inputs, training=None):
        noisy_inputs, noise_true = self.add_noise(inputs, self.step)
        
        drop_input = self.dropout(noisy_inputs)
        features = self.encoder(drop_input)
        noise_pred = self.decoder(features)
        
        loss = self.mse_loss(noise_true, noise_pred)
        self.add_loss(loss)

        if training:
            self.step += self.rate

        return noisy_inputs - noise_pred
    
auto_encoder = CustomAutoencoder(encoder, decoder, 3.0, 0.5, 1/5000)

# Custom Activation to maintain zero centered with max standard deviation of 3
def tanh3(x):
    return tf.nn.tanh(x) * 3.5
keras.utils.get_custom_objects().update({'tanh3': tanh3})

def create_classifier(encoder, classes):
    e_input_chans = encoder.input_shape[-1]
    return Sequential([
        # Expansion Block
        Sequential([
            Conv1D(e_input_chans//4, kernel, padding='same'), Activation(act),
            Conv1D(e_input_chans//2, kernel, padding='same'), Activation(act), 
            Conv1D(e_input_chans//1, kernel, padding='same'), Activation('linear'),
            SpatialDropout1D(0.2)
        ]),

        encoder,

        # Classification Block
        Sequential([
            SpatialAttention(classes, 5),
            GlobalAveragePooling1D(),
            Dropout(0.5),
            Dense(classes, activation='softmax', kernel_regularizer='l2')
        ])
    ])

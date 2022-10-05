import tensorflow.keras as keras
from tensorflow.keras import layers

from scale_crosscorrelation import *
from scale_space_layers import *


class MaxProjection(tf.keras.layers.Layer):
    """ Performs max projection on the signal on the semigroup of scales and translations """
    def call(self, inputs):
        return tf.reduce_max(inputs, 3)


class UpsampleEquivariant(layers.Layer):
    """ Layer that upsamples a function on the semigroup of scales and translations using bilinear interpolation"""
    def __init__(self, size=(2, 2), **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def build(self, input_shape):
        self.n_scales = input_shape[3]
        self.n_channels = input_shape[4]

        super().build(input_shape)

    def call(self, inputs):
        shape = tf.shape(inputs)
        out = tf.reshape(inputs, [shape[0], shape[1], shape[2], shape[3] * shape[4]])
        h = self.size[0] * shape[1]
        w = self.size[1] * shape[2]
        out = tf.image.resize(out, [h, w])
        return tf.reshape(out, [shape[0], h, w, self.n_scales, self.n_channels])

    def get_config(self):
        config = super(UpsampleEquivariant, self).get_config()
        config.update({'size': self.size})
        return config


def scale_conv_block(x, filters, n_scales, scale_dim, name, batchnorm_momentum):
    """ Block of two scale cross-correlations """
    x = ScaleConv(filters, (3, 3, 1), n_scales, name=f'{name}_1_sconv')(x)
    x = layers.BatchNormalization(momentum=batchnorm_momentum, name=f'{name}_1_bn')(x)
    x = layers.Activation('relu', name=f'{name}_1_relu')(x)

    x = ScaleConv(filters, (3, 3, scale_dim), n_scales, name=f'{name}_2_sconv')(x)
    x = layers.BatchNormalization(momentum=batchnorm_momentum, name=f'{name}_2_bn')(x)
    x = layers.Activation('relu', name=f'{name}_2_relu')(x)
    return x


def upsample_block(x, res, filters, n_scales, scale_dim, name, batchnorm_momentum):
    """ Block that performs scale cross-correlations followed by upsample and concatenation with the skip connection """
    x = ScaleConv(filters, (3, 3, 1), n_scales, name=f'{name}_1_sconv')(x)
    x = layers.BatchNormalization(momentum=batchnorm_momentum, name=f'{name}_1_bn')(x)
    x = layers.Activation('relu', name=f'{name}_1_relu')(x)

    x = ScaleConv(filters, (3, 3, scale_dim), n_scales, name=f'{name}_2_sconv')(x)
    x = layers.BatchNormalization(momentum=batchnorm_momentum, name=f'{name}_2_bn')(x)
    x = layers.Activation('relu', name=f'{name}_2_relu')(x)

    # x = layers.Lambda(upsample, name=f'{name}_upsample', output_shape=x.shape)(x)
    x = UpsampleEquivariant((2,2), name=f'{name}_upsample')(x)

    x = layers.Concatenate(axis=-1, name=f'{name}_skip')([x, res])
    x = layers.Conv3D(filters, 1, name=f'{name}_3_conv')(x)
    x = layers.BatchNormalization(momentum=batchnorm_momentum, name=f'{name}_3_bn')(x)
    x = layers.Activation('relu', name=f'{name}_3_relu')(x)

    return x


def pool(x, method, name):
    """ Pooling dependent on the choosen method """
    if method == 'Quadratic':
        return ScaleQuadraticDilation(.5, strides=(2, 2), max_width=3, name=name)(x)
    elif method == 'Quadratic2':
        return ScaleQuadraticDilation(1., strides=(2, 2), max_width=3, name=name)(x)
    elif method == 'Flat':
        return ScaleFlatDilation(1, strides=(2, 2), name=name)(x)
    else:
        return layers.MaxPool3D((1, 1, 1), strides=(2, 2, 1), name=name)(x)


# returns an object of the keras Model class which computes the Scale-Equivariant U-Net
# img_size: a tuple (height, width, channels) specifying the dimensions of the input images
#   (height and width can be None for undetermined values)
# lifting: a keras Layer object that computes the lifting of the model
# n_scales: number of scales used before truncatin. The approximate equivariance will be empirically verified ing
# scale_dim: size of the filters in the scale coordinate
# n_filters: filters in the first layer of the model
#   filters in the proceeding layers depends on that value
# n_classes: number of classes, that is, the number of channels of the last layer
# depth: number of poolings/upsamplings, height
# dropout: float between 0 and 1, dropout rate
# activation: activation of the last layer
# pooling: string specifying the pooling used
#   Quadratic for quadratic dilation
#   Flat for flat dilation / maxpool
#   None or anything else for no pooling

def seunet(img_size, lifting, n_scales, scale_dim, n_filters, n_classes, depth=4, dropout=0, activation=None,
               pooling_method='Quadratic', batchnorm_momentum=.98):
    """ Function to instantiate a SEUNet method """
    inputs = keras.Input(shape=img_size)

    x = IdLifting(n_scales, name='id_lifting')(inputs)
    x = lifting(x)

    x = ScaleConv(n_filters, (3, 3, 1), n_scales, name='block1_sconv')(x)
    x = layers.BatchNormalization(momentum=batchnorm_momentum, name='block1_bn')(x)
    x = layers.Activation('relu', name='block1_relu')(x)

    res = list()

    for i, filters in enumerate([n_filters * (2 ** i) for i in range(depth)]):
        x = scale_conv_block(x, filters, n_scales, scale_dim, f'down{i+1}', batchnorm_momentum)
        res.append(x)  # Set aside next residual
        x = pool(x, pooling_method, f'down{i+1}_pool')

    for i, filters in enumerate([n_filters * (2 ** i) for i in range(depth, 0, -1)]):
        residual = res.pop()
        x = upsample_block(x, residual, filters, n_scales, scale_dim, f'up{i+1}', batchnorm_momentum)

    x = scale_conv_block(x, n_filters, n_scales, 1, 'block_out', batchnorm_momentum)

    if dropout > 0:
        x = ScaleDropout(dropout, name='dropout')(x)
    x = MaxProjection(name='proj')(x)


    outputs = layers.Conv2D(n_classes, 1, activation=activation, padding='same', name='classifier')(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

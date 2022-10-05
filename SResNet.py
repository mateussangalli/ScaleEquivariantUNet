import tensorflow.keras as keras
from tensorflow.keras import layers

from scale_crosscorrelation import *
from scale_space_layers import *


class AvgProjection(layers.Layer):
    def call(self, inputs):
        out = tf.reduce_mean(inputs, 3)
        return out


class PadAdd(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shapes):
        s1 = input_shapes[0][-1]
        s2 = input_shapes[1][-1]
        if s1 > s2:
            self.which_pad = 2
        elif s2 > s1:
            self.which_pad = 1
        else:
            self.which_pad = 0
        self.pad_size = np.abs(s1 - s2)

        super().build(input_shapes)

    def call(self, inputs):
        a, b = inputs
        if self.which_pad == 2:
            return a + tf.pad(b, [[0, 0], [0, 0], [0, 0], [0, 0], [0, self.pad_size]])
        elif self.which_pad == 1:
            return b + tf.pad(a, [[0, 0], [0, 0], [0, 0], [0, 0], [0, self.pad_size]])
        else:
            return a + b


def scale_conv_block(x, filters, n_scales, scale_dim, strides=(1, 1), name='block', batchnorm_momentum=.99):
    x = ScaleConv(filters, (3, 3, 1), n_scales, name=f'{name}_1_sconv')(x)
    x = layers.BatchNormalization(momentum=batchnorm_momentum, name=f'{name}_1_bn')(x)
    x = layers.Activation('relu', name=f'{name}_1_relu')(x)

    x = ScaleConv(filters, (3, 3, scale_dim), n_scales, strides=strides, name=f'{name}_2_sconv')(x)
    x = layers.BatchNormalization(momentum=batchnorm_momentum, name=f'{name}_2_bn')(x)
    return x


def sresnet(img_size, lifting, N, n_scales, scale_dim, n_classes, batchnorm_momentum=.99, dropout=0):
    """ Function to instantiate a SResNet model """
    inputs = keras.Input(shape=img_size)

    x = IdLifting(n_scales, name='id_lifting')(inputs)
    x = lifting(x)

    x = scale_conv_block(x, N, n_scales, scale_dim, (2, 2), 'conv1', batchnorm_momentum)

    x0 = layers.MaxPool3D((1, 1, 1), strides=(2, 2, 1), name='strides1')(x)
    x1 = scale_conv_block(x, 2 * N, n_scales, scale_dim, (2, 2), 'conv2', batchnorm_momentum)
    x = PadAdd(name='add1')([x0, x1])

    x1 = scale_conv_block(x, 4 * N, n_scales, 1, (1, 1), 'conv3', batchnorm_momentum)
    x = PadAdd(name='add2')([x, x1])
    x0 = layers.MaxPool3D((1, 1, 1), strides=(2, 2, 1), name='strides3')(x)
    x1 = scale_conv_block(x, 4 * N, n_scales, scale_dim, (2, 2), 'conv4', batchnorm_momentum)
    x = PadAdd(name='add3')([x0, x1])

    x1 = scale_conv_block(x, 8 * N, n_scales, 1, (1, 1), 'conv5', batchnorm_momentum)
    x = PadAdd(name='add4')([x, x1])

    x1 = scale_conv_block(x, 8 * N, n_scales, 1, (1, 1), 'conv6', batchnorm_momentum)
    x = PadAdd(name='add5')([x, x1])
    x1 = scale_conv_block(x, 8 * N, n_scales, 1, (1, 1), 'conv7', batchnorm_momentum)
    x = PadAdd(name='add6')([x, x1])
    x1 = scale_conv_block(x, 8 * N, n_scales, scale_dim, (1, 1), 'conv8', batchnorm_momentum)
    x = PadAdd(name='add7')([x, x1])

    x = scale_conv_block(x, 8 * N, n_scales, scale_dim, (1, 1), 'conv9', batchnorm_momentum)
    if dropout > 0:
        x = layers.Dropout(dropout, noise_shape=(1, 1, 1, None, 1), name='dropout')(x)
    x = AvgProjection(name='proj')(x)

    outputs = layers.Conv2D(n_classes, 1, padding='same', name='classifier')(x)
    outputs = layers.UpSampling2D((8, 8))(outputs)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

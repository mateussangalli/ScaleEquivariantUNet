import tensorflow.keras as keras
from tensorflow.keras import layers

BN_MOMENTUM = 0.9


def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_1_conv')(x)
    # x = layers.LayerNormalization(name=f'{name}_1_ln')(x)
    x = layers.BatchNormalization(name=f'{name}_1_bn')(x)
    x = layers.Activation('relu', name=f'{name}_1_relu')(x)

    x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_2_conv')(x)
    # x = layers.LayerNormalization(name=f'{name}_2_ln')(x)
    # x = layers.experimental.SyncBatchNormalization(name=f'{name}_2_bn')(x)
    x = layers.BatchNormalization(name=f'{name}_2_bn')(x)
    x = layers.Activation('relu', name=f'{name}_2_relu')(x)
    return x


def upsample_block(x, res, filters, name):
    x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_1_conv')(x)
    # x = layers.LayerNormalization(name=f'{name}_1_ln')(x)
    x = layers.BatchNormalization(name=f'{name}_1_bn')(x)
    x = layers.Activation('relu', name=f'{name}_1_relu')(x)

    x = layers.Conv2D(filters, 3, padding='same', name=f'{name}_2_conv')(x)
    # x = layers.LayerNormalization(name=f'{name}_2_ln')(x)
    x = layers.BatchNormalization(name=f'{name}_2_bn')(x)
    x = layers.Activation('relu', name=f'{name}_2_relu')(x)

    x = layers.UpSampling2D(2, name=f'{name}_upsample')(x)

    x = layers.Concatenate(axis=-1, name=f'{name}_skip')([x, res])
    x = layers.Conv2D(filters, 1, name=f'{name}_3_conv')(x)
    # x = layers.LayerNormalization(name=f'{name}_3_ln')(x)
    x = layers.BatchNormalization(name=f'{name}_3_bn')(x)
    x = layers.Activation('relu', name=f'{name}_3_relu')(x)

    return x


def unet(img_size, n_filters, n_classes, depth=4, activation='softmax'):
    inputs = keras.Input(shape=img_size)

    # Entry block
    x = layers.Conv2D(n_filters, 3, padding='same', name='block1_conv')(inputs)
    # x = layers.LayerNormalization(name='block1_ln')(x)
    x = layers.BatchNormalization(name='block1_bn')(x)
    x = layers.Activation('relu', name='block1_relu')(x)

    res = list()

    for i, filters in enumerate([n_filters * (2 ** i) for i in range(depth)]):
        x = conv_block(x, filters, f'down{i+1}')
        res.append(x)  # Set aside next residual
        x = layers.MaxPooling2D(name=f'down{i+1}_pool')(x)

    for i, filters in enumerate([n_filters * (2 ** i) for i in range(depth, 0, -1)]):
        residual = res.pop()
        x = upsample_block(x, residual, filters, f'up{i+1}')

    x = conv_block(x, n_filters, name='block_out')

    outputs = layers.Conv2D(n_classes, 1, padding='same', activation=activation, name='classifier')(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

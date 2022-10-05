import numpy as np
import tensorflow as tf


def pad_edges(x, pads):
    (pad_u, pad_d), (pad_l, pad_r) = pads
    x = tf.concat([tf.tile(x[:, 0, tf.newaxis, :, :], (1, pad_u, 1, 1)), x], 1)
    x = tf.concat([x, tf.tile(x[:, -1, tf.newaxis, :, :], (1, pad_d, 1, 1))], 1)
    x = tf.concat([tf.tile(x[:, :, 0, tf.newaxis, :], (1, 1, pad_l, 1)), x], 2)
    x = tf.concat([x, tf.tile(x[:, :, -1, tf.newaxis, :], (1, 1, pad_r, 1))], 2)
    return x


def pad_min(x, pads):
    min_value = tf.reduce_min(x)
    paddings = [(0,0), *pads, (0,0)]
    x = tf.pad(x, paddings, constant_values=min_value)
    return x


class MaxProjection(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, 3)


class AvgProjection(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, 3)


class Simplex(tf.keras.constraints.Constraint):
    """
    Constraint to Values
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, x):
        x = tf.clip_by_value(x, 0, float('inf'))
        x = x / tf.reduce_sum(x, 3, keepdims=True)
        return x


class ClipValuesBetween(tf.keras.constraints.Constraint):
    """
    Constraint to Values
    """

    def __init__(self, min_value=0, max_value=1):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, x):
        return tf.clip_by_value(x, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


class IdLifting(tf.keras.layers.Layer):
    def __init__(self, n_scales, **kwargs):
        super().__init__(**kwargs)
        self.n_scales = n_scales

    def call(self, x):
        x = tf.expand_dims(x, 3)
        return tf.tile(x, tf.constant([1, 1, 1, self.n_scales, 1], dtype=tf.int32))

    def get_config(self):
        config = super().get_config().copy()
        config.update({'n_scales': self.n_scales})
        return config


class ScaleGaussian(tf.keras.layers.Layer):
    def __init__(self, zero_scale=1., max_width=5, strides=(1, 1), base=2, start_at_one=False, trainable_params=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.strides = strides
        self.zero_scale = zero_scale
        self.start_at_one = start_at_one
        self.trainable_params = trainable_params
        self.max_width = max_width
        self.base = base

    def build(self, input_shape):
        self.n_scales = input_shape[-2]
        if self.start_at_one:
            self.scales = [float((self.base ** i) - 1) * self.zero_scale * 2 for i in range(self.n_scales)]
        else:
            self.scales = [float((self.base ** i) - 1) * self.zero_scale * 2 for i in range(1, self.n_scales + 1)]

        vmin = ((2 * 1.645 * self.zero_scale) ** 2) / self.max_width ** 2
        vmax = ((2 * 1.645 * self.zero_scale) ** 2)

        self.coef = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer='ones',
            constraint=ClipValuesBetween(vmin, vmax),
            trainable=True, name='scale_coef')

        # widths = [tf.cast(self.max_width*scale, tf.int32) for scale in self.scales]
        # filters = [tf.cast(tf.linspace(-w, w, 2*w+1), tf.float32) for w in widths]
        # self.filters = [-((k[:,tf.newaxis]**2)/(scale**2)) for k,scale in zip(filters,self.scales)]
        # self.widths = widths
        widths = [tf.cast(self.max_width * (self.base ** i), tf.int32) for i in range(self.n_scales)]
        filters = [tf.cast(tf.linspace(-w, w, 2 * w + 1), tf.float32) for w in widths]
        self.filters = [-((k[:, tf.newaxis] ** 2) / (scale ** 2)) for k, scale in zip(filters, self.scales)]
        self.widths = widths

    def call(self, x):
        out = []
        for i in range(self.n_scales):
            if i == 0 and self.start_at_one:
                out.append(x[..., 0, :])
                continue
            k = tf.exp(self.filters[i] * self.coef)
            k = k / tf.reduce_sum(k, 0, keepdims=True)
            kx = k[:, tf.newaxis, :, tf.newaxis]
            ky = k[tf.newaxis, :, :, tf.newaxis]
            x0 = x[..., i, :]
            s = self.widths[i]
            pad = [s, s]
            pads = [pad, pad]
            x0 = pad_edges(x0, pads)

            x0 = tf.nn.depthwise_conv2d(x0, kx, (1, self.strides[0], 1, 1), padding='VALID')
            x0 = tf.nn.depthwise_conv2d(x0, ky, (1, 1, self.strides[1], 1), padding='VALID')
            out.append(x0)

        return tf.stack(out, -2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'zero_scale': self.zero_scale,
                       'strides': self.strides,
                       'max_width': self.max_width,
                       'base': self.base,
                       'start_at_one': self.start_at_one})
        return config


class ScaleQuadraticDilation(tf.keras.layers.Layer):
    def __init__(self, zero_scale=1., max_width=5, strides=(1, 1), base=2, start_at_one=False, **kwargs):
        super().__init__(**kwargs)

        self.strides = strides
        self.zero_scale = zero_scale
        self.start_at_one = start_at_one
        self.max_width = max_width
        self.base = base

    def build(self, input_shape):
        self.n_scales = input_shape[-2]
        if self.start_at_one:
            self.scales = [float((self.base ** i) - 1) * self.zero_scale * 2 for i in range(self.n_scales)]
        else:
            self.scales = [float((self.base ** i) - 1) * self.zero_scale * 2 for i in range(1, self.n_scales + 1)]

        vmin = (4 * self.zero_scale ** 2) / float(self.max_width ** 2)
        vmax = 4 * self.zero_scale ** 2

        self.coef = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer='ones',
            constraint=ClipValuesBetween(vmin, vmax),
            trainable=True, name='scale_coef')

        widths = [tf.cast(self.max_width * (self.base ** i), tf.int32) for i in range(self.n_scales)]
        filters = [tf.cast(tf.linspace(-w, w, 2 * w + 1), tf.float32) for w in widths]
        self.filters = [-((k[:, tf.newaxis] ** 2) / (scale ** 2)) for k, scale in zip(filters, self.scales)]
        self.widths = widths

    def call(self, x):
        out = []
        for i in range(self.n_scales):
            if i == 0 and self.start_at_one:
                out.append(x[..., 0, :])
                continue
            k = self.filters[i] * self.coef
            kx = k[:, tf.newaxis, :]
            ky = k[tf.newaxis, :, :]
            x0 = x[..., i, :]
            s = self.widths[i]
            pad = [s, s]
            pads = [pad, pad]
            x0 = pad_min(x0, pads)

            x0 = tf.nn.dilation2d(x0, kx, (1, self.strides[0], 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_out = tf.nn.dilation2d(x0, ky, (1, 1, self.strides[1], 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            out.append(x_out)

        return tf.stack(out, -2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'zero_scale': self.zero_scale,
                       'strides': self.strides,
                       'max_width': self.max_width,
                       'base': self.base,
                       'start_at_one': self.start_at_one})
        return config


class ScaleErosionPlusDilation(tf.keras.layers.Layer):
    def __init__(self, zero_scale=1., strides=(1, 1), base=2, initial_alpha=.5, **kwargs):
        super().__init__(**kwargs)

        self.strides = strides
        self.zero_scale = zero_scale
        self.initial_alpha = initial_alpha
        self.base = base

    def build(self, input_shape):
        self.n_scales = input_shape[-2]
        self.scales = [max(int((self.base ** i) * self.zero_scale + .5), 1) for i in range(self.n_scales)]

        self.alpha = self.add_weight(
            shape=(1, 1, 1, input_shape[-1]),
            initializer=tf.keras.initializers.Constant(self.initial_alpha),
            trainable=True, name='alpha')

    def call(self, x):
        out = []
        for i in range(self.n_scales):
            if i == 0:
                out.append(x[..., 0, :])
                continue
            xd = tf.nn.max_pool2d(x[..., i, :], self.scales[i], self.strides, 'SAME', 'NHWC')
            xe = -tf.nn.max_pool2d(-x[..., i, :], self.scales[i], self.strides, 'SAME', 'NHWC')
            x0 = self.alpha * xd + (1. - self.alpha) * xe
            out.append(x0)
        return tf.stack(out, -2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'zero_scale': self.zero_scale,
                       'initial_alpha': self.initial_alpha,
                       'base': self.base,
                       'strides': self.strides})
        return config


class ScaleFlatDilation(tf.keras.layers.Layer):
    def __init__(self, zero_scale=1., strides=(1, 1), base=2, **kwargs):
        super().__init__(**kwargs)

        self.strides = strides
        self.zero_scale = zero_scale
        self.base = base

    def build(self, input_shape):
        self.n_scales = input_shape[-2]
        self.scales = [max(int((self.base ** i) * self.zero_scale + .5), 1) for i in range(self.n_scales)]

    def call(self, x):
        out = []
        for i in range(self.n_scales):
            x0 = tf.nn.max_pool2d(x[..., i, :], self.scales[i], self.strides, 'SAME', 'NHWC')
            out.append(x0)
        return tf.stack(out, -2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'zero_scale': self.zero_scale,
                       'base': self.base,
                       'strides': self.strides})
        return config


class ScaleLasryLions(tf.keras.layers.Layer):
    def __init__(self, zero_scale=.25, max_width=5, strides=(1, 1), base=2, start_at_one=False, **kwargs):
        super().__init__(**kwargs)

        self.strides = strides
        self.zero_scale = zero_scale
        self.start_at_one = start_at_one
        self.max_width = max_width
        self.base = base

    def build(self, input_shape):
        self.n_scales = input_shape[-2]
        if self.start_at_one:
            self.scales = [float((self.base ** i) - 1) * self.zero_scale for i in range(self.n_scales)]
        else:
            self.scales = [float((self.base ** i) - 1) * self.zero_scale for i in range(1, self.n_scales + 1)]

        self.coef = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer='ones',
            constraint=ClipValuesBetween(1 / float(self.max_width ** 2), float('inf')),
            trainable=True, name='scale_coef')

        self.c = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            initializer=tf.keras.initializers.Constant(2.),
            constraint=ClipValuesBetween(1., float('inf')),
            trainable=True, name='c')

        self.alpha = self.add_weight(
            shape=(1, 1, 1, input_shape[-1]),
            initializer=tf.keras.initializers.Constant(.5),
            constraint=ClipValuesBetween(0., 1.),
            trainable=True, name='alpha')

        widths = [tf.cast(self.max_width * scale, tf.int32) for scale in self.scales]
        filters = [tf.cast(tf.linspace(-w, w, 2 * w + 1), tf.float32) for w in widths]
        self.filters = [-((k[:, tf.newaxis] ** 2) / (scale ** 2)) for k, scale in zip(filters, self.scales)]
        self.widths = widths

    def call(self, x):

        out = []
        for i in range(self.n_scales):
            if i == 0 and self.start_at_one:
                out.append(x[..., 0, :])
                continue
            k = self.filters[i] * self.coef
            kx = k[:, tf.newaxis, :]
            ky = k[tf.newaxis, :, :]
            x0 = x[..., i, :]
            s = self.widths[i]
            pad = [2 * s, 2 * s]
            pads = [pad, pad]
            x0 = pad_edges(x0, pads)

            x_clo = tf.nn.dilation2d(x0, kx, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_clo = tf.nn.dilation2d(x_clo, ky, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_clo = tf.nn.erosion2d(x_clo, kx * self.c, (1, self.strides[0], 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_clo = tf.nn.erosion2d(x_clo, ky * self.c, (1, 1, self.strides[1], 1), 'VALID', 'NHWC', (1, 1, 1, 1))

            x_open = tf.nn.erosion2d(x0, kx, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_open = tf.nn.erosion2d(x_open, ky, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_open = tf.nn.dilation2d(x_open, kx * self.c, (1, self.strides[0], 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_open = tf.nn.dilation2d(x_open, ky * self.c, (1, 1, self.strides[1], 1), 'VALID', 'NHWC', (1, 1, 1, 1))

            x_out = self.alpha * x_clo + (1 - self.alpha) * x_open
            out.append(x_out)

        return tf.stack(out, -2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'zero_scale': self.zero_scale,
                       'strides': self.strides,
                       'max_width': self.max_width,
                       'base': self.base,
                       'start_at_one': self.start_at_one})
        return config


class ScaleGaussianLasryLions(tf.keras.layers.Layer):
    def __init__(self, zero_scale=.25, max_width=5, strides=(1, 1), base=2, start_at_one=False, **kwargs):
        super().__init__(**kwargs)

        self.strides = strides
        self.zero_scale = zero_scale
        self.start_at_one = start_at_one
        self.max_width = max_width
        self.base = base

    def build(self, input_shape):
        self.n_scales = input_shape[-2]
        if self.start_at_one:
            self.scales = [float((self.base ** i) - 1) * self.zero_scale for i in range(self.n_scales)]
        else:
            self.scales = [float((self.base ** i) - 1) * self.zero_scale for i in range(1, self.n_scales + 1)]

        self.coef_morpho = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer='ones',
            constraint=ClipValuesBetween(1 / float(self.max_width ** 2), float('inf')),
            trainable=True, name='scale_coef_morpho')

        self.coef_gaussian = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer='ones',
            constraint=ClipValuesBetween(1 / float(self.max_width ** 2), float('inf')),
            trainable=True, name='scale_coef_gaussian')

        self.c = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            initializer=tf.keras.initializers.Constant(2.),
            constraint=ClipValuesBetween(1., float('inf')),
            trainable=True, name='c')

        self.alpha = self.add_weight(
            shape=(1, 1, 1, input_shape[-1], 3),
            initializer=tf.keras.initializers.Constant(1 / 3),
            constraint=Simplex(),
            trainable=True, name='alpha')

        widths = [tf.cast(self.max_width * scale, tf.int32) for scale in self.scales]
        filters = [tf.cast(tf.linspace(-w, w, 2 * w + 1), tf.float32) for w in widths]
        self.filters = [-((k[:, tf.newaxis] ** 2) / (scale ** 2)) for k, scale in zip(filters, self.scales)]
        self.widths = widths

    def call(self, x):

        out = []
        for i in range(self.n_scales):
            if i == 0 and self.start_at_one:
                out.append(x[..., 0, :])
                continue
            k = self.filters[i] * self.coef_morpho
            kx = k[:, tf.newaxis, :]
            ky = k[tf.newaxis, :, :]
            kg = tf.exp(self.filters[i] * self.coef_gaussian)
            kgx = k[:, tf.newaxis, :, tf.newaxis]
            kgy = k[tf.newaxis, :, :, tf.newaxis]
            x0 = x[..., i, :]
            s = self.widths[i]
            pad = [s, s]
            paddings = [pad, pad]
            x0 = pad_edges(x0, paddings)
            x_blur = tf.nn.depthwise_conv2d(x0, kgx, (1, self.strides[0], 1, 1), padding='VALID')
            x_blur = tf.nn.depthwise_conv2d(x_blur, kgy, (1, 1, self.strides[1], 1), padding='VALID')

            x_clo = tf.nn.dilation2d(x0, kx, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_clo = tf.nn.dilation2d(x_clo, ky, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_clo = tf.nn.erosion2d(pad_edges(x_clo, paddings), kx * self.c, (1, self.strides[0], 1, 1), 'VALID',
                                    'NHWC', (1, 1, 1, 1))
            x_clo = tf.nn.erosion2d(x_clo, ky * self.c, (1, 1, self.strides[1], 1), 'VALID', 'NHWC', (1, 1, 1, 1))

            x_open = tf.nn.erosion2d(x0, kx, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_open = tf.nn.erosion2d(x_open, ky, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1))
            x_open = tf.nn.dilation2d(pad_edges(x_open, paddings), kx * self.c, (1, self.strides[0], 1, 1), 'VALID',
                                      'NHWC', (1, 1, 1, 1))
            x_open = tf.nn.dilation2d(x_open, ky * self.c, (1, 1, self.strides[1], 1), 'VALID', 'NHWC', (1, 1, 1, 1))

            x_out = self.alpha[..., 0] * x_clo + self.alpha[..., 1] * x_open + self.alpha[..., 1] * x_blur
            out.append(x_out)

        return tf.stack(out, -2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'zero_scale': self.zero_scale,
                       'strides': self.strides,
                       'max_width': self.max_width,
                       'base': self.base,
                       'start_at_one': self.start_at_one})
        return config


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage.io import imread

    n_scales = 5

    im = imread('poivrons.png')
    im = im.astype(np.float32) / 255
    im = im[np.newaxis, ...]
    print(im.max())
    print(im.shape)

    inputs = tf.keras.layers.Input((None, None, 3))
    x = IdLifting(n_scales)(inputs)
    x = ScaleGaussian(.5)(x)
    model1 = tf.keras.models.Model(inputs, x)

    inputs = tf.keras.layers.Input((None, None, 3))
    x = IdLifting(n_scales)(inputs)
    x = ScaleQuadraticDilation(.5)(x)
    model2 = tf.keras.models.Model(inputs, x)

    im_g = model1(im)[0, ...]
    im_d = model2(im)[0, ...]

    fig = plt.figure()
    for i in range(n_scales):
        plt.subplot(2, n_scales, i + 1)
        plt.imshow(im_g[:, :, i, :])
        plt.title(f'gaussian {i}')
    for i in range(n_scales):
        plt.subplot(2, n_scales, n_scales + i + 1)
        plt.imshow(im_d[:, :, i, :])
        plt.title(f'dilation {i}')
    plt.show()

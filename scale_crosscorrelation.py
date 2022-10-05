import tensorflow as tf
from tensorflow.keras import backend as K


class ScaleDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.seed = None

    def build(self, input_shape):
        self.noise_shape = (None, 1, 1, input_shape[3], input_shape[4])

    def _get_noise_shape(self, inputs):
        # Subclasses of `Dropout` may implement `_get_noise_shape(self, inputs)`,
        # which will override `self.noise_shape`, and allows for custom noise
        # shapes with dynamically sized inputs.
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = tf.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return tf.convert_to_tensor(noise_shape)

    def call(self, x, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            return (1 - self.rate) * tf.nn.dropout(
                x,
                noise_shape=self._get_noise_shape(x),
                seed=self.seed,
                rate=self.rate)

        output = tf.cond(tf.constant(bool(training), dtype=tf.bool), dropped_inputs, lambda: tf.identity(x))
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({'rate': self.rate})
        return config


class ScaleConv(tf.keras.layers.Layer):
    def __init__(self, n_out, kernel_size, n_scales_out, use_bias=True, base=2, strides=(1, 1),
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        # kernel size: tuple (Kr, Kc, Ks)
        # Kr and Kc must be odd
        super().__init__(**kwargs)

        self.n_out = n_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.n_scales_out = n_scales_out
        self.base = base
        self.dilations = [base ** i for i in range(n_scales_out)]
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        # input_shape: tuple (B,H,W,S,C)
        # B, H, W, C are the usual dimensions
        # S is the scale dimension
        self.n_scales_in = input_shape[-2]
        n_channels = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], self.kernel_size[2] * input_shape[-1], self.n_out),
            initializer=self.kernel_initializer,
            trainable=True, name='weight')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.n_out,),
                                        initializer=self.bias_initializer, trainable=True, name='bias')

        k = tf.range(self.kernel_size[2])
        k = 2 ** (3 * k)
        k = tf.reshape(k, [1, 1, 1, -1, 1])
        self.k = 1 / tf.cast(k, tf.float32)

    def compute_scale_conv(self, x):
        shape = tf.shape(x)
        out = list()
        for i in range(self.n_scales_out):
            d = self.dilations[i]
            smax = min(self.n_scales_in, i + self.kernel_size[2])
            x0 = x[..., i:smax, :]
            if smax - i < self.kernel_size[2]:
                x0 = tf.concat([x0] + [x0[..., -1:, :]] * (self.kernel_size[2] + i - smax), -2)

            x0 = tf.reshape(x0 * self.k, [shape[0], shape[1], shape[2], -1])
            sr = d * (self.kernel_size[0] // 2)
            sc = d * (self.kernel_size[1] // 2)
            pad_r = [sr, sr]
            pad_c = [sc, sc]
            paddings = [(0, 0), pad_r, pad_c, (0, 0)]
            x0 = tf.pad(x0, paddings)

            x0 = tf.nn.conv2d(x0, self.kernel, (1, *self.strides, 1), 'VALID', dilations=d)
            out.append(x0)
        return tf.stack(out, 3)

    def call(self, x):
        out = self.compute_scale_conv(x)
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)

        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({'n_out': self.n_out,
                       'kernel_size': self.kernel_size,
                       'n_scales_out': self.n_scales_out,
                       'use_bias': self.use_bias,
                       'base': self.base,
                       'kernel_initializer': self.kernel_initializer,
                       'bias_initializer': self.bias_initializer,
                       'strides': self.strides})
        return config

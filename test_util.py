import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

IMG_SIZE = (512, 512)

def rescaled_preproc_func(scale):
    @tf.function
    def rescaled_preproc(x, y):
        height = int(IMG_SIZE[0] * scale)
        height = 16 * (height // 16)
        width = int(IMG_SIZE[1] * scale)
        width = 16 * (width // 16)
        img_size = (height, width)
        x = tf.image.resize(x, img_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        y = tf.image.resize(tf.expand_dims(y, 2), img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return x, y[:, :, 0]

    return rescaled_preproc

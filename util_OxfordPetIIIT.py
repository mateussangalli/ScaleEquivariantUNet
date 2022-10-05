import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_RATIO = .8
IMG_SIZE = (224, 224)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], IMG_SIZE)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], IMG_SIZE, method='nearest')
    input_mask = tf.cast(input_mask, tf.int32)
    return normalize(input_image, input_mask)


def get_train_and_val():
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir='./')
    TRAIN_LENGTH = info.splits['train'].num_examples
    n_train = int(TRAIN_RATIO * TRAIN_LENGTH)
    n_val = TRAIN_LENGTH - n_train

    train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE).take(n_train)
    val_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE).skip(n_train)
    return train_images, val_images, n_train, n_val


def rescaled_preproc_func(scale):
    @tf.function
    def rescaled_preproc(x, y):
        height = int(IMG_SIZE[0] * scale)
        height = 16 * (height // 16)
        width = int(IMG_SIZE[1] * scale)
        width = 16 * (width // 16)
        img_size = (height, width)
        x = tf.image.resize(x, img_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        y = tf.image.resize(y, img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return x, y
    return rescaled_preproc


def get_test():
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir='./')
    length = info.splits['test'].num_examples

    ds = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds, length


class MyMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)


class MaxProjection(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, 3)


class AvgProjection(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, 3)

def normalized_mse(a, b):
    return tf.reduce_mean((a - b) ** 2) / (tf.reduce_mean(a ** 2) + 1e-10)


def dataset_mse(model, original_dataset, transformed_dataset):
    mses = list()
    for (x, y), (x2, y2) in zip(original_dataset, transformed_dataset):
        a = model(x)
        b = model(x2)
        b = tf.image.resize(b, IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mses.append(normalized_mse(a, b))
    return np.array(mses).mean()


def dataset_cosine_similarity(model, original_dataset, transformed_dataset):
    cos_metric = tf.keras.metrics.CosineSimilarity()
    for (x, y), (x2, y2) in zip(original_dataset, transformed_dataset):
        a = model(x)
        b = model(x2)
        b = tf.image.resize(b, IMG_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        cos_metric.update_state(a, b)
    return cos_metric.result()


def dataset_consistency(model, original_dataset, transformed_dataset):
    consistency_metric = tf.keras.metrics.Accuracy()
    for (x, y), (x2, y2) in zip(original_dataset, transformed_dataset):
        a = tf.argmax(model(x), axis=3)
        b = model(x2)
        b = tf.image.resize(b, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        b = tf.argmax(b, axis=3)
        consistency_metric.update_state(np.ones_like(a), tf.cast(a == b, tf.int32))
    return consistency_metric.result()

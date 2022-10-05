import tensorflow as tf
from imgaug import augmenters as iaa
import wandb
from wandb.keras import WandbCallback
import numpy as np
from util_DIC_C2DH_HeLa import CustomIoU, WeightDecayScheduler
from SEUNet import seunet
from SResNet import sresnet
from unet import unet
from scale_space_layers import ScaleGaussian
import os
import tensorflow_addons as tfa
from load_util import load_train, load_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='which model to train')
parser.add_argument('--pooling', type=str, help='which pooling to use', default='Quadratic')
parser.add_argument('--n_scales', type=int, help='number of scales', default=4)
parser.add_argument('--scale_dim', type=int, help='scale dimension of the filters', default=1)
parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-4)
parser.add_argument('--dropout', type=float, default=.25)
parser.add_argument('--id', type=int, default=0)
parser.add_argument('--scale_aug', type=bool, default=False)
parser.add_argument('--scale_aug_range', type=float, default=2.)
parser.add_argument('--datadir', type=str, default='./')
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

saved_models_dir = 'saved_models'
datadir = args.data_dir

BATCH_SIZE = 1
VAL_FREQ = 5
MAX_EPOCHS = args.epochs
N_CLASSES = 2


if args.model == 'seunet':
    model_name = f'seunet_{args.pooling}_sd{args.scale_dim}_dropout{int(args.dropout * 100)}'
elif args.model == 'sresnet':
    model_name = f'sresnet_sd{args.scale_dim}_dropout{int(args.dropout * 100)}'
elif args.model == 'unet':
    model_name = f'unet'
else:
    raise ValueError('model not recognized!')


def get_model():
    if args.model == 'seunet':
        return seunet((None, None, 1),
                      ScaleGaussian(.25, start_at_one=True),
                      args.n_scales,
                      args.scale_dim,
                      64,
                      N_CLASSES,
                      4,
                      args.dropout,
                      pooling_method=args.pooling)
    elif args.model == 'sresnet':
        return sresnet((None, None, 1),
                       ScaleGaussian(.25, start_at_one=True),
                       64,
                       args.n_scales,
                       args.scale_dim,
                       N_CLASSES,
                       args.dropout)
    elif args.model == 'unet':
        return unet((None, None, 1),
                    64,
                    N_CLASSES,
                    4,
                    None)
    else:
        raise ValueError('model not recognized!')

n_val = 10
x_train, y_train = load_train(datadir)
x_train = x_train.astype(np.float32)[..., np.newaxis] / 255
y_train = (y_train > 0).astype(np.int32)

n_train = x_train.shape[0] - n_val
x_val = x_train[-n_val:, ...]
y_val = y_train[-n_val:, ...]
x_train = x_train[:n_train, ...]
y_train = y_train[:n_train, ...]


aug = iaa.Sequential([
    iaa.Rotate((-10, 10), mode='reflect'),
    iaa.Fliplr(.5),
    iaa.Flipud(.5),
    iaa.ElasticTransformation(),
    iaa.LinearContrast((0.9, 1.1), per_channel=True),
], random_order=True)

scale_aug = iaa.Sequential([
    iaa.Resize((1 / args.scale_aug_range, args.scale_aug_range)),
    iaa.CropToFixedSize(x_train.shape[1], x_train.shape[2]),
    iaa.PadToFixedSize(x_train.shape[1], x_train.shape[2])
], random_order=False)


def augment(x, y):
    x, y = aug(image=x, segmentation_maps=y[np.newaxis, :, :, np.newaxis])
    if args.scale_aug:
        x, y = scale_aug(image=x, segmentation_maps=y)
    return x, y[0, :, :, 0]


@tf.function
def aug_tf(x, y):
    x, y = tf.numpy_function(augment, [x, y], [tf.float32, tf.int32])
    x = tf.ensure_shape(x, (x_train.shape[1], x_train.shape[2], 1))
    y = tf.ensure_shape(y, (x_train.shape[1], x_train.shape[2]))
    return x, y



ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(n_train)
ds_train = ds_train.map(aug_tf, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
STEPS_PER_EPOCH = int(np.ceil(n_train / BATCH_SIZE))

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.cache()
ds_val = ds_val.batch(1)
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
VAL_STEPS_PER_EPOCH = n_val


def schedule(epoch, lr):
    if epoch > 1:
        return lr * np.exp(-np.log(10) / 100)
    return lr



lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)
wd_callback = WeightDecayScheduler(schedule, verbose=1)



model = get_model()
model.summary()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tfa.optimizers.AdamW(weight_decay=args.weight_decay,
                                             learning_rate=1e-3),
              metrics=[CustomIoU(N_CLASSES, [1]), 'accuracy'])
model.fit(ds_train, steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=ds_val, validation_steps=VAL_STEPS_PER_EPOCH,
          epochs=MAX_EPOCHS,
          callbacks=[lr_callback, wd_callback],
          verbose=2,
          validation_freq=VAL_FREQ)
if args.scale_aug:
    scale_aug_str = '_scale_aug'
    tmp = f'{args.scale_aug_range:.2f}'.replace('.', 'p')
    scale_aug_str += tmp
else:
    scale_aug_str = ''
model.save(os.path.join('saved_models', model_name+scale_aug_str+f'_id{args.id}'))

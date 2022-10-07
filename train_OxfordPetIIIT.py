import os
import tensorflow_datasets as tfds
from SEUNet import seunet
from unet import unet
from SResNet import sresnet
from imgaug import augmenters as iaa
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from scale_space_layers import ScaleGaussian
from util_OxfordPetIIIT import get_train_and_val, MyMeanIoU


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='which model to train')
parser.add_argument('--start', type=int, help='index of the first model', default=0)
parser.add_argument('--end', type=int, help='index after the last model', default=1)
parser.add_argument('--pooling', type=str, help='which pooling to use', default='Quadratic')
parser.add_argument('--n_scales', type=int, help='number of scales', default=4)
parser.add_argument('--scale_dim', type=int, help='scale dimension of the filters', default=1)
parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.)
parser.add_argument('--dropout', type=float, default=.25)
parser.add_argument('--scale_aug', type=bool, default=False)
parser.add_argument('--scale_aug_range', type=float, default=2.)
parser.add_argument('--epochs', type=int, default=300)

args = parser.parse_args()

saved_models_dir = 'saved_models'
IMG_SIZE = (224, 224)
CROP_SIZE = (112, 112)
EPOCHS =  args.epochs
BATCH_SIZE = 8



if args.model == 'seunet':
    model_name = f'seunet_{args.pooling}_sd{args.scale_dim}_OxfordPetIIIT_dropout{int(args.dropout * 100)}'
elif args.model == 'sresnet':
    model_name = f'sresnet_sd{args.scale_dim}_OxfordPetIIIT_dropout{int(args.dropout * 100)}'
elif args.model == 'unet':
    model_name = f'unet_OxfordPetIIIT'
else:
    raise ValueError('model not recognized!')


def get_model():
    if args.model == 'seunet':
        return seunet((None, None, 3),
                      ScaleGaussian(.25, start_at_one=True),
                      args.n_scales,
                      args.scale_dim,
                      16,
                      3,
                      4,
                      args.dropout,
                      pooling_method=args.pooling)
    elif args.model == 'sresnet':
        return sresnet((None, None, 3),
                       ScaleGaussian(.25, start_at_one=True),
                       16,
                       args.n_scales,
                       args.scale_dim,
                       3,
                       args.dropout)
    elif args.model == 'unet':
        return unet((None, None, 3),
                    16,
                    3,
                    4,
                    None)
    else:
        raise ValueError('model not recognized!')


if args.scale_aug:
    aug = iaa.Sequential([
        iaa.Rotate((-10, 10), mode='reflect'),
        iaa.Fliplr(.5),
        iaa.LinearContrast((0.9, 1.1), per_channel=True),
        # iaa.CropToFixedSize(CROP_SIZE[1], CROP_SIZE[0])
    ], random_order=True)
else:
    aug = iaa.Sequential([
        iaa.Rotate((-10, 10), mode='reflect'),
        iaa.Fliplr(.5),
        iaa.LinearContrast((0.9, 1.1), per_channel=True),
        iaa.CropToFixedSize(CROP_SIZE[1], CROP_SIZE[0])
    ], random_order=True)

scale_aug = iaa.Sequential([
    iaa.Resize((1 / args.scale_aug_range, args.scale_aug_range)),
    iaa.CropToFixedSize(CROP_SIZE[0], CROP_SIZE[1]),
    iaa.PadToFixedSize(CROP_SIZE[0], CROP_SIZE[1])
], random_order=False)


def augment(x, y):
    x, y = aug(image=x, segmentation_maps=y[np.newaxis, ...])
    if args.scale_aug:
        x, y = scale_aug(image=x, segmentation_maps=y)

    return x, y[0, :, :, :].astype(np.int32)


@tf.function
def aug_tf(x, y):
    x, y = tf.numpy_function(augment, [x, y], [tf.float32, tf.int32])
    x = tf.ensure_shape(x, (CROP_SIZE[0], CROP_SIZE[1], 3))
    y = tf.ensure_shape(y, (CROP_SIZE[0], CROP_SIZE[1], 1))
    return x, y


train_images, val_images, n_train, n_val = get_train_and_val()
STEPS_PER_EPOCH = int(np.ceil(n_train / BATCH_SIZE))
VAL_STEPS_PER_EPOCH = n_val

train_batches = (
    train_images
    .cache()
    .shuffle(n_train)
    .map(aug_tf)
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE))

val_batches = val_images.cache().batch(1)


config = vars(args)
for i in range(args.start, args.end):

    model = get_model()
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tfa.optimizers.AdamW(args.weight_decay, 1e-3), metrics=[MyMeanIoU(3)])

    cb_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=30, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-6
    )


    model_history = model.fit(train_batches, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_data=val_batches,
                              validation_steps=VAL_STEPS_PER_EPOCH,
                              callbacks=[cb_lr],
                              verbose=2)

    scale_aug_str = ''
    if args.scale_aug:
        scale_aug_str += f'aug{int(args.scale_aug_range * 10)}'
    model.save(os.path.join(saved_models_dir,
                            f'{model_name+scale_aug_str}_rep{i}'))

import tensorflow as tf
import tensorflow_datasets as tfds
from imgaug import augmenters as iaa
import numpy as np
from util_DIC_C2DH_HeLa import CustomIoU, WeightDecayScheduler
from test_util import rescaled_preproc_func
from SEUNet import seunet
from SResNet import sresnet
from unet import unet
from scale_space_layers import ScaleGaussian
import os
import tensorflow_addons as tfa
from imgaug import augmenters as iaa
from load_util import load_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help='which model to train')
parser.add_argument('--n_models', type=int, help='How many instances of the model to test', default='3')
parser.add_argument('--pooling', type=str, help='which pooling to use', default='Quadratic')
parser.add_argument('--n_scales', type=int, help='number of scales', default=4)
parser.add_argument('--scale_dim', type=int, help='scale dimension of the filters', default=1)
parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-5)
parser.add_argument('--dropout', type=float, default=.25)
parser.add_argument('--scale_aug', type=bool, default=False)
parser.add_argument('--scale_aug_range', type=float, default=2.)
parser.add_argument('--datadir', type=str, default='./')
args = parser.parse_args()

datadir = args.datadir
saved_models_dir = 'saved_models'
results_dir = 'results'
patch_size = 448
extra_size = 224

if args.model == 'seunet':
    model_name = f'seunet_{args.pooling}_sd{args.scale_dim}_dropout{int(args.dropout * 100)}'
elif args.model == 'sresnet':
    model_name = f'sresnet_sd{args.scale_dim}_dropout{int(args.dropout * 100)}'
elif args.model == 'unet':
    model_name = f'unet'
else:
    raise ValueError('model not recognized!')


def pred_big(model, image, patch_size, extra_size):
    shape = image.shape
    image = iaa.PadToMultiplesOf(patch_size, patch_size, position='right-bottom')(image=image[0, :, :, :])
    image = image[np.newaxis, ...]
    shape2 = image.shape
    n_patches = int(shape2[1]) // patch_size

    actual_patch_size = patch_size + 2 * extra_size
    image = tf.image.extract_patches(image,
                                     (1, actual_patch_size, actual_patch_size, 1),
                                     (1, patch_size, patch_size, 1),
                                     (1, 1, 1, 1),
                                     'SAME')
    image = image[0, :n_patches, :n_patches, :]
    image = tf.reshape(image, (n_patches, n_patches, actual_patch_size, actual_patch_size, 1))
    out = np.zeros((shape2[1], shape2[2], 2), np.float32)
    for i in range(n_patches):
        for j in range(n_patches):
            tmp = tf.expand_dims(image[i, j, :, :, :], 0)
            tmp = model(tmp)
            tmp = np.array(tmp)
            out[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :] = tmp[0,
                                                                                               extra_size:-extra_size,
                                                                                               extra_size:-extra_size,
                                                                                               :]
    return out[:shape[1], :shape[2], :]


x_test, y_test = load_test(datadir)
x_test = x_test.astype(np.float32)[..., np.newaxis] / 255
y_test = (y_test > 0).astype(np.int32)
VAL_STEPS_PER_EPOCH = x_test.shape[0]
scales = 2 ** np.linspace(-3, 2, 11).astype(np.float32)

if args.scale_aug:
    scale_aug_str = '_scale_aug'
    tmp = f'{args.scale_aug_range:.2f}'.replace('.', 'p')
    scale_aug_str += tmp
else:
    scale_aug_str = ''
ious_reps = list()
for i in range(args.n_models):
    custom_objects = {'CustomIoU': CustomIoU(2, [1]), 'AdamW': tfa.optimizers.AdamW}
    # model = tf.keras.models.load_model(model_name, custom_objects=custom_objects)
    model = tf.keras.models.load_model(os.path.join(saved_models_dir, model_name + scale_aug_str + f'_id{i}'),
                                       custom_objects=custom_objects)
    ious = list()

    for scale in scales:
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        ds_test = ds_test.map(rescaled_preproc_func(scale), num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.cache()
        ds_test = ds_test.batch(1)
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
        if scale <= 1.:
            _, iou, _ = model.evaluate(ds_test, verbose=2)
        else:
            iou_metric = CustomIoU(2, [1])
            for x, y in ds_test.as_numpy_iterator():
                pred = pred_big(model, x, patch_size, extra_size)
                iou_metric.update_state(y, pred[np.newaxis, ...])
            iou = iou_metric.result()
            print(iou)
        ious.append(iou)

    ious = np.array(ious)
    print(ious)
    ious_reps.append(ious)

ious = np.stack(ious_reps, 0)

np.save(os.path.join(results_dir, model_name + scale_aug_str + '.npy'), ious)

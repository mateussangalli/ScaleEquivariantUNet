import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from scale_space_layers import ScaleGaussian
from util_OxfordPetIIIT import get_test, MyMeanIoU, rescaled_preproc_func
from util_OxfordPetIIIT import dataset_cosine_similarity, dataset_consistency, dataset_mse


import wandb
from wandb.keras import WandbCallback

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
args = parser.parse_args()

saved_models_dir = 'saved_models'
results_dir = 'results'
IMG_SIZE = (224, 224)
CROP_SIZE = (112, 112)
EPOCHS = 300
BATCH_SIZE = 8

if args.model == 'seunet':
    model_name = f'seunet_{args.pooling}_sd{args.scale_dim}_OxfordPetIIIT_dropout{int(args.dropout * 100)}'
elif args.model == 'sresnet':
    model_name = f'sresnet_sd{args.scale_dim}_OxfordPetIIIT_dropout{int(args.dropout * 100)}'
elif args.model == 'unet':
    model_name = f'unet_OxfordPetIIIT'
else:
    raise ValueError('model not recognized!')

scale_aug_str = ''
if args.scale_aug:
    scale_aug_str += f'_aug{int(args.scale_aug_range * 10)}'

model_name += scale_aug_str



test_images, n_test = get_test()

test_batches = test_images.cache().batch(1)


config = vars(args)
scales = 2 ** np.linspace(-2, 2, 17)
accs = list()
mses = list()
cons = list()
sims = list()
for i in range(args.start, args.end):
    custom_objects = {'MyMeanIoU': MyMeanIoU(3)}
    model = tf.keras.models.load_model(os.path.join(saved_models_dir, f'{model_name}_rep{i}'),
                                       custom_objects=custom_objects)
    accs_tmp = list()
    mses_tmp = list()
    cons_tmp = list()
    sims_tmp = list()
    for s in scales:
        ds_new = test_images.map(rescaled_preproc_func(s), num_parallel_calls=tf.data.AUTOTUNE).batch(1)
        _, acc = model.evaluate(ds_new, verbose=2)
        con = dataset_consistency(model, test_batches, ds_new)
        # mse = dataset_mse(model, test_batches, ds_new)
        # sim = dataset_cosine_similarity(model, test_batches, ds_new)
        accs_tmp.append(acc)
        cons_tmp.append(con)
        # mses_tmp.append(mse)
        # sims_tmp.append(sim)
    accs.append(accs_tmp)
    cons.append(cons_tmp)
    # mses.append(mses_tmp)
    # sims.append(sims_tmp)

accs = np.array(accs)
cons = np.array(cons)
# mses = np.array(mses)
# sims = np.array(sims)


np.save(os.path.join(results_dir, f'{model_name}_acc'), accs)
np.save(os.path.join(results_dir, f'{model_name}_con'), cons)
# np.save(os.path.join(results_dir, f'{model_name}_mse'), mses)
# np.save(os.path.join(results_dir, f'{model_name}_sim'), sims)

print('accuracies')
print(accs)
print('consistencies')
print(cons)
# print('mses')
# print(mses)
# print('cosine similarities')
# print(sims)

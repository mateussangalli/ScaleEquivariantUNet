from skimage.io import imread
import numpy as np
import os


def load_train(data_dir):
    images_dir = '01'
    labels_dir = '01_ST/SEG'
    images = list()
    ground_truths = list()
    for filename in os.listdir(os.path.join(data_dir, images_dir)):
        path_im = os.path.join(data_dir, images_dir, filename)
        path_gt = os.path.join(data_dir, labels_dir, 'man_seg' + filename[1:])
        images.append(imread(path_im))
        ground_truths.append(imread(path_gt))
    return np.stack(images, 0), np.stack(ground_truths, 0)



def load_test(data_dir):
    images_dir = '02'
    labels_dir = '02_ST/SEG'
    images = list()
    ground_truths = list()
    for filename in os.listdir(os.path.join(data_dir, images_dir)):
        path_im = os.path.join(data_dir, images_dir, filename)
        path_gt = os.path.join(data_dir, labels_dir, 'man_seg' + filename[1:])
        images.append(imread(path_im))
        ground_truths.append(imread(path_gt))
    return np.stack(images, 0), np.stack(ground_truths, 0)


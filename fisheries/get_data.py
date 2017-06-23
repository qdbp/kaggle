from functools import lru_cache, partial
from glob import glob
import os
import os.path as osp
from threading import Lock

from joblib import Memory
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from scipy.misc import comb

from keras.preprocessing.image import ImageDataGenerator as IDG, load_img
from keras.preprocessing.image import img_to_array
from skimage.io import imread, imsave
from skimage.transform import resize
# from PIL import Image

C, W, H = 3, 50, 50

TRAIN_DIR = '/home/main/programming/kaggle/fisheries/data/train/'
TEST_DIR = '/home/main/programming/kaggle/fisheries/data/test/'

ORDER = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
NDIM = len(ORDER)


idg = IDG(
   # featurewise_center=True,
   # featurewise_std_normalization=True,
   rotation_range=5.,
   width_shift_range=0.05,
   height_shift_range=0.05,
   zoom_range=(0.95, 1/0.95),
   channel_shift_range=0.5,
   horizontal_flip=True,
   vertical_flip=True,
   fill_mode='constant',
   cval=0,
)


os.makedirs('/tmp/fisheries', exist_ok=True)
mem = Memory(cachedir='/tmp/fisheries', verbose=0)


@mem.cache
def read_imagefile(fn, c, w, h):
    img = load_img(fn, keep_aspect_ratio=True, grayscale=(c == 1),
                   target_size=(w, h))
    return img_to_array(img)


def load_train_files():
    l = 0
    for o in ORDER:
        l += len(glob(osp.join(TRAIN_DIR, o, '*jpg')))

    y = np.zeros((l, NDIM), dtype=np.uint8)
    X = np.zeros((l, C, H, W), dtype=np.float32)

    ix = 0
    for ox, o in enumerate(ORDER):
        fns = sorted(glob(osp.join(TRAIN_DIR, o, '*jpg')))
        for fx, fn in enumerate(fns):
            X[ix] = read_imagefile(fn, C, W, H)[:]
            y[ix, ox] = 1
            ix += 1

    assert ix == l

    return X, y


cmode = 'grayscale' if C == 1 else 'rgb'


def load_test_data():
    test_files = sorted(glob(osp.join(TEST_DIR, '*.jpg')))
    l = len(test_files)
    assert l > 0

    X = np.zeros((l, C, W, H))

    for fx, fn in enumerate(test_files):
        X[fx] = idg.standardize(read_imagefile(fn, C, W, H)[:])

    return test_files, X


if __name__ == '__main__':
    print(idg.mean, idg.std)
    load_test_data()

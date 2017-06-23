from glob import glob
import os.path as osp
import pickle

import numpy as np
import numpy.random as npr
import scipy.linalg as nlg
from skimage.io import imread
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from PIL import Image

from qqq.qlog import get_logger

LOG = get_logger(__file__)

LABEL_FN = './train.csv'
TRAIN_FNS = sorted(glob('./train-tif/train_*.tif'))
TEST_FNS = sorted(glob('./test-tif/*.tif'))
N_TRAIN = len(TRAIN_FNS)
N_TEST = len(TEST_FNS)


def get_label_rows():
    with open(LABEL_FN, 'r') as f:
        for rx, row in enumerate(f):
            row = row.strip()
            if rx == 0:
                continue
            yield tuple(row.split(','))


def validate_alignment():
    for fn, labrow in zip(TRAIN_FNS, get_label_rows()):
        check = int(labrow[0].split('_')[1])
        check_str = f'{check:>05d}'
        assert check_str in fn, f'{labrow[0]} not in {fn}'


def mk_labels():
    y_fn = './npy/y_train.npy'
    lab_fn = './binarizer.p'

    if not osp.isfile(lab_fn) or not osp.isfile(y_fn):
        validate_alignment()

        labelss = [row[1].split(' ') for row in get_label_rows()]

        mlb = MultiLabelBinarizer()
        out = mlb.fit_transform(labelss)

        LABELS = mlb.classes_

        Y_TRAIN = np.memmap(
            y_fn, dtype=np.uint8, shape=(N_TRAIN, len(LABELS)), mode='w+')
        Y_TRAIN[:] = out.astype(np.uint8)[:]

        with open(lab_fn, 'wb') as f:
            pickle.dump(LABELS, f)

    with open(lab_fn, 'rb') as f:
        LABELS = pickle.load(f)

    Y_TRAIN = np.memmap(
        y_fn, dtype=np.uint8, shape=(N_TRAIN, len(LABELS)), mode='r')

    return LABELS, Y_TRAIN


def load_images(width, kind='train'):

    if kind != 'train' and kind != 'test':
        raise ValueError('images must be train or test')

    arr_fn = f'./npy/{kind}_{width}.npy'
    n = N_TRAIN if kind == 'train' else N_TEST
    fns = TRAIN_FNS if kind == 'train' else TEST_FNS
    shape = (n, width, width, 4)

    if not osp.isfile(arr_fn):
        LOG.info(f'did not find {arr_fn}, creating')
        ARR = np.memmap(arr_fn, dtype=np.float16, shape=shape, mode='w+')

        for i in tqdm(range(n)):
            im_arr = imread(fns[i]).astype(np.float32) / 10000.
            for j in range(4):
                ARR[i, :, :, j] =\
                    Image.fromarray(im_arr[:, :, j])\
                    .resize((width, width), Image.ANTIALIAS)

    return np.memmap(arr_fn, dtype=np.float16, shape=shape, mode='r')


def get_mean_std(x_arr, axis=(0, 1, 2)):
    m = x_arr.mean(axis=axis, dtype=np.float64)
    s = x_arr.std(axis=axis, dtype=np.float64)

    return m, s


def get_whitening_matrix(x_arr, mean):

    x_flat = x_arr.reshape(-1, x_arr.shape[-1])
    ixes = npr.permutation(len(x_flat))[:10_000_000]
    x_flat = x_flat[ixes]

    W = nlg.inv(nlg.sqrtm(np.cov((x_flat - mean).T)))

    return W.T



if __name__ == '__main__':
    validate_alignment()

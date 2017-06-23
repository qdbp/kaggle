# flake8: noqa f401
from functools import partial
import glob
import multiprocessing as mp
import os
import os.path as osp
import pickle
import re
import sys
from itertools import cycle

import click as clk
import keras.initializers as kri
import keras.models as krm
import keras.layers as kr
import keras.regularizers as krr
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import scipy.linalg as nlg
from skimage.io import imread
from sklearn.decomposition import IncrementalPCA
import sklearn.metrics as skm
from tqdm import tqdm

from qqq.keras_util import ValLossPP, apply_layers
from qqq.keras_util import get_callbacks, f2_crossentropy
from qqq.ml import generate_batches, batch_transformer as bt, gen_mlc_weights
from qqq.ml import dict_tv_split, apply_bts, get_dict_data_len
from qqq.ml import mk_bin_from_mlc
from qqq.image import color_zca
from qqq.image import cartesian_shift, subpixel_shift, image_flip
from qqq.image import chunk_image, chunk_image_shape
from qqq.np import tandem_shuffle
from qqq.qlog import get_logger
from qqq.util import kws, kwsift, pickled

from preprocess import mk_labels, load_images
from preprocess import get_whitening_matrix, get_mean_std
from preprocess import TRAIN_FNS, TEST_FNS

LOG = get_logger(__file__)

LABELS, Y_TRAIN = mk_labels()
NL = len(LABELS)


def load_data(width):

    X_TRAIN = load_images(width, kind='train')
    X_TEST = load_images(width, kind='test')

    M_TRAIN, S_TRAIN = pickled(f'm_s_{width}.p', get_mean_std, X_TRAIN)

    W_ZCA = pickled(f'zca_{width}.p', get_whitening_matrix, X_TRAIN, M_TRAIN)

    return X_TRAIN, X_TEST, M_TRAIN, W_ZCA


def grind_threshs_fbeta(yt, yp, beta=2):
    threshs = []
    for i in tqdm(range(yp.shape[1])):
        f1s = []
        ths = []
        for j in tqdm(np.linspace(0, 1, num=100)):
            y_cand = (yp[:, i] > j).astype(np.uint8)
            f1s.append(skm.fbeta_score(yt[:, i], y_cand, beta=beta))
            ths.append(j)

        th_ix = np.argmax(f1s)
        threshs.append(ths[th_ix])

    yp_cat = (yp > np.array(threshs)).astype(np.uint8)
    print(skm.classification_report(Y_TRAIN, yp_cat))
    return yp_cat, threshs


def grind_threshs_stats(yt, yp):
    fracs = np.sum(yt, axis=0) / len(yt)

    threshs = []
    for i in tqdm(range(yp.shape[1])):
        cur_loss = np.inf
        threshs.append(0.5)
        for j in tqdm(np.linspace(0, 1, num=1000)):
            frac = (yp[:, i] > j).sum(dtype=np.float64) / len(yp)
            loss = np.abs(frac - fracs[i])
            if loss < cur_loss:
                cur_loss = loss
                threshs[i] = j

    yp_cat = (yp > np.array(threshs)).astype(np.uint8)
    print(skm.classification_report(Y_TRAIN, yp_cat))
    return yp_cat, threshs


def mk_chunked_128(channels):

    i = kr.Input((64, 16, 16, channels), name='x0')

    conv_stack = [
        kr.GaussianNoise(0.05),
        kr.Conv2D(16, 3, strides=1, activation='relu', name='chunk128_c0'),
        kr.Conv2D(24, 3, strides=1, activation='relu', name='chunk128_c1'),
        kr.Conv2D(32, 3, strides=1, activation='relu', name='chunk128_c2'),
        kr.Conv2D(48, 3, strides=1, activation='relu', name='chunk128_c3'),
        kr.Conv2D(64, 3, strides=1, activation='relu', name='chunk128_c4'),
        kr.Conv2D(96, 3, strides=1, activation='relu', name='chunk128_c5'),
        kr.Conv2D(128, 3, strides=1, activation='relu', name='chunk128_c6'),
        kr.Conv2D(256, 2, strides=1, activation='relu', name='chunk128_c7'),
        kr.Flatten(),
        kr.Dense(256, activation='elu', name='chunk128_d0'),
        kr.Dropout(0.5),
        kr.Dense(128, activation='elu', name='chunk128_d1'),
    ]

    h_td = apply_layers(i, conv_stack, td=True)

    lstm_stack = [
        kr.LSTM(256, name='chunk128_lstm'),
        kr.Dense(64, activation='elu', name='chunk128_dl0'),
        kr.Dense(Y_TRAIN.shape[1], activation='sigmoid', name='labs'),
    ]

    y = apply_layers(h_td, lstm_stack)

    m = krm.Model(inputs=[i], outputs=[y], name='base_conv')
    m.compile(
        loss=f2_crossentropy, optimizer='nadam',
        metrics=['binary_accuracy'])

    return m


def mk_hybrid_128(channels):

    i_16 = kr.Input((16, 16, channels), name='x_16')
    i_128 = kr.Input((64, 16, 16, channels), name='x')

    conv16_stack = [
        kr.GaussianNoise(0.05),
        kr.Conv2D(16, 3, strides=1, activation='relu', name='chunk128_c0'),
        kr.Conv2D(24, 3, strides=1, activation='relu', name='chunk128_c1'),
        kr.Conv2D(32, 3, strides=1, activation='relu', name='chunk128_c2'),
        kr.Conv2D(48, 3, strides=1, activation='relu', name='chunk128_c3'),
        kr.Conv2D(64, 3, strides=1, activation='relu', name='chunk128_c4'),
        kr.Conv2D(96, 3, strides=1, activation='relu', name='chunk128_c5'),
        kr.Conv2D(128, 3, strides=1, activation='relu', name='chunk128_c6'),
        kr.Conv2D(256, 2, strides=1, activation='relu', name='chunk128_c7'),
        kr.Flatten(),
        kr.Dense(256, activation='elu', name='chunk128_d0'),
        kr.Dropout(0.5),
        kr.Dense(128, activation='elu', name='chunk128_d1'),
    ]

    conv_td_stack = [
        kr.GaussianNoise(0.05),
        kr.Conv2D(16, 3, strides=1, activation='relu'),
        kr.Conv2D(24, 3, strides=1, activation='relu'),
        kr.Conv2D(32, 3, strides=1, activation='relu'),
        kr.Conv2D(48, 3, strides=1, activation='relu'),
        kr.Conv2D(64, 3, strides=1, activation='relu'),
        kr.Conv2D(96, 3, strides=1, activation='relu'),
        kr.Conv2D(128, 3, strides=1, activation='relu'),
        kr.Conv2D(256, 2, strides=1, activation='relu'),
        kr.Flatten(),
        kr.Dense(256, activation='elu'),
        kr.Dropout(0.5),
        kr.Dense(128, activation='elu'),
    ]

    h_td = apply_layers(i_128, conv_td_stack, td=True)

    lstm_stack = [
        kr.LSTM(256),
        kr.Dense(128, activation='elu'),
    ]

    h_128 = apply_layers(h_td, lstm_stack)
    h_16 = apply_layers(i_16, conv16_stack)

    cat = kr.Concatenate()([h_16, h_128])

    head = [
        kr.Dense(256, activation='elu'),
        kr.Dropout(0.5),
        kr.Dense(128, activation='elu'),
        kr.Dense(Y_TRAIN.shape[1], activation='sigmoid', name='labs'),
    ]

    y = apply_layers(cat, head)

    m = krm.Model(inputs=[i_16, i_128], outputs=[y], name='hybrid')
    m.compile(
        loss='binary_crossentropy', optimizer='nadam',
        metrics=['binary_accuracy'])

    return m

@kws
def mk_conv_64(*, channels):

    i = kr.Input((64, 64, channels), name='x0')
    
    cc_stack = [
        kr.GaussianNoise(0.025),
        kr.Conv2D(10, 1),
        kr.LeakyReLU(alpha=0.5),
        kr.Conv2D(2),
    ]

    h = apply_layers(i, cc_stack)

    conv_stack = [
        kr.GaussianNoise(0.025),
        kr.Conv2D(64, 3, strides=1, activation='elu', ),
        kr.Conv2D(128, 3, strides=1, activation='elu', ),
        kr.Conv2D(256, 3, strides=1, activation='elu', ),
        kr.Conv2D(384, 3, strides=2, activation='elu', ),
    ]

    h = apply_layers(i, conv_stack)
    gp0 = kr.GlobalMaxPool2D()(h)
    ap0 = kr.GlobalAveragePooling2D()(h)

    conv_stack_2 = [
        kr.Conv2D(384, 3, strides=1, activation='elu', ),
        kr.Conv2D(384, 3, strides=1, activation='elu', ),
        kr.Conv2D(384, 3, strides=2, activation='elu', ),
    ]

    conv_stack_3 = [
        kr.Conv2D(198, 3, strides=1, activation='elu', ),
        kr.Conv2D(198, 3, strides=1, activation='elu', ),
        kr.Conv2D(198, 3, strides=1, activation='elu', ),
        kr.Conv2D(198, 3, strides=2, activation='elu', ),
        kr.Flatten(),
    ]

    h = apply_layers(h, conv_stack_2)
    h = apply_layers(h, conv_stack_3)

    head = [
        kr.Dense(512, activation='elu'),
        kr.Dropout(0.5),
        kr.Dense(256, activation='elu'),
        kr.Dropout(0.5),
        kr.Dense(128, activation='elu'),
        kr.Dense(NL, activation='sigmoid', name='labs'),
    ]

    cat = kr.Concatenate()([h, gp0, ap0])
    y = apply_layers(cat, head)

    m = krm.Model(inputs=[i], outputs=[y], name='base_conv')
    m.compile(
        loss=f2_crossentropy, optimizer='adam',
        metrics=['binary_accuracy'])

    return m

@kws
def mk_conv_32(*, channels):
    i = kr.Input((32, 32, 4), name='x0')

    cc_stack = [
        # kr.BatchNormalization(),
        kr.GaussianNoise(0.025),
        kr.Conv2D(10, 1, name='cconv_0'),
        kr.LeakyReLU(alpha=0.4),
        kr.Conv2D(channels, 1, name='cconv_1'),
        kr.LeakyReLU(alpha=0.4),
    ]

    h = apply_layers(i, cc_stack)

    conv_stack_0 = [
        kr.GaussianNoise(0.025),
        kr.Conv2D(64, 3, activation='relu'),
        kr.Conv2D(128, 3, activation='relu'),
        kr.Conv2D(256, 3, activation='relu'),
        kr.Conv2D(256, 3, strides=2, activation='relu'),
    ]
   
    h = apply_layers(h, conv_stack_0)
    ga0 = kr.GlobalAveragePooling2D()(h)
    gm0 = kr.GlobalMaxPooling2D()(h)
    
    conv_stack_1 = [
        kr.Conv2D(196, 3, activation='relu'),
        kr.Conv2D(196, 3, strides=2, activation='relu'),
        kr.Conv2D(196, 3, activation='relu'),
        kr.Flatten(),
    ]

    h = apply_layers(h, conv_stack_1)
    cat = kr.Concatenate()([h, gm0, ga0])

    head = [
        kr.Dropout(0.5),
        kr.Dense(512, activation='elu'),
        kr.Dropout(0.5),
        kr.Dense(256, activation='elu'),
        kr.Dropout(0.5),
        kr.Dense(Y_TRAIN.shape[1], activation='sigmoid', name='labs'),
    ]

    y = apply_layers(cat, head)

    m = krm.Model(inputs=[i], outputs=[y], name='conv_32')
    m.compile(
        loss=f2_crossentropy, optimizer='adam',
        metrics=['binary_accuracy'])

    return m


def mk_conv_16(channels, loss_weights=None, mode='joint'):

    i = kr.Input((16, 16, channels), name='x0')

    conv_stack = [
        kr.GaussianNoise(0.05),
        kr.Conv2D(32, 3, strides=1, activation='relu', name='conv16_c0'),
        kr.Conv2D(128, 3, strides=1, activation='relu', name='conv16_c1'),
        kr.Conv2D(196, 3, strides=1, activation='relu', name='conv16_c2'),
        kr.Conv2D(256, 3, strides=1, activation='relu', name='conv16_c3'),
        kr.Conv2D(196, 3, strides=1, activation='relu', name='conv16_c4'),
        kr.Conv2D(196, 3, strides=1, activation='relu', name='conv16_c5'),
        kr.Conv2D(196, 3, strides=1, activation='relu', name='conv16_c6'),
        kr.Conv2D(128, 2, strides=1, activation='relu', name='conv16_c7'),
    ]

    h = apply_layers(i, conv_stack)

    head = [
        kr.Flatten(),
        kr.Dense(128, activation='elu', name='conv16_d0'),
        kr.Dropout(0.5),
        kr.Dense(128, activation='elu', name='conv16_d1'),
    ]

    h = apply_layers(h, head)

    if mode == 'joint':
        ys = [kr.Dense(NL, activation='sigmoid', name='labs')(h)]
        loss = f2_crossentropy
        metrics = ['binary_accuracy']

    elif mode in {'binary', 'indiv'}:
        loss = f2_crossentropy
        metrics = ['categorical_accuracy']
        
        if mode == 'binary':
            ys = [
                kr.Dense(2, activation='softmax', name=f'lab_{lab}')(h)
                for lab in LABELS]
        else:
            ys = [kr.Dense(2, activation='softmax', name='lab')(h)]

    else:
        raise ValueError(f'invalid mode {mode}')

    m = krm.Model(inputs=[i], outputs=ys, name=f'conv_16_{mode}')
    m.compile(loss=loss, metrics=metrics, optimizer='adam')

    return m


MODELS = {
    'conv': {
        16: mk_conv_16,
        32: mk_conv_32,
        64: mk_conv_64,
    },
    'chunk': {
        128: mk_chunked_128,
    },
    'hybrid': {
        128: mk_hybrid_128,
    }
}


def mk_model_name(make, width, variant, *, channels, mode='joint', label=None):
    base = f'{make}{width}_{variant}'
    if mode == 'binary':
        base += '_binary'
    elif mode == 'indiv':
        if label is None:
            raise ValueError('need to give label name for individual models')
        base += f'_indiv_{label}'

    return base


def train_model(
        make, width, variant, w_zca, mu, x, y, *, f_train=0.85, bs,
        sample_weight=None, load=False, mode='joint', **kw):

    if mode != 'joint':
        y_bin = mk_bin_from_mlc(y['labs'])
        if mode == 'indiv':
            yds = [{'lab': yb} for yb in y_bin]
        elif mode == 'binary':
            yds = [{f'lab_{lab}': yb for lab, yb in zip(LABELS, y_bin)}]
    else:
        yds = [y]

    for lab, yd in zip(LABELS, yds):

        name = mk_model_name(
            make, width, variant, label=lab, mode='mode',
            **kwsift(kw, mk_model_name))

        LOG.info(f'training model {name}')

        if not load:
            m = MODELS[make][width](mode=mode, **kw)
        else:
            m = krm.load_model(f'./weights/{name}.hdf5')

        print(m.summary())

        (x_t, y_t), (x_v, y_v) = dict_tv_split(x, yd, f_train=f_train, seed=55)

        t_gen = generate_batches(x_t, y=y_t, bs=bs, balance=mode == 'indiv')
        v_gen = generate_batches(x_v, y=y_v, bs=bs)

        bts = [
            bt(partial(color_zca, mu=mu, W=w_zca), inplace=True),
            bt(image_flip, inplace=True, train_only=True),
            bt(subpixel_shift, inplace=True, train_only=True),
        ]

        if make == 'chunk' or make == 'hybrid':
            bts.append(bt(
                partial(chunk_image, chunk_size=16), in_keys=['x'],
                inplace=False,
                get_shape=partial(chunk_image_shape, chunk_size=16),
            ))

        t_gen = apply_bts(t_gen, bts, train=True)
        v_gen = apply_bts(v_gen, bts, train=False)

        m.fit_generator(
            t_gen, 100, validation_data=v_gen,
            validation_steps=30, epochs=50, verbose=0,
            callbacks=get_callbacks(name, **kwsift(kw, get_callbacks)),
            **kwsift(kw, m.fit_generator),
        )


def predict_model(make, width, variant, w_zca, mu,
                  x, augments=10, bag=False, bs=32, **kw):

    name = mk_model_name(make, width, variant, **kwsift(kw, mk_model_name))
    LOG.info(f'predicting model {name}')

    fns = glob.glob(f'./weights/{name}_bag*.hdf5')
    if not fns:
        fns = [f'./weights/{name}.hdf5']
    else:
        LOG.info(f'found {len(fns)} bag subclassifiers')

    do_augment = bool(augments)
    if not do_augment:
        augments = 1

    l = get_dict_data_len(x)
    n = augments * l
    out = None

    for fn in tqdm(fns):

        m = krm.load_model(
            fn, custom_objects={'f2_crossentropy': f2_crossentropy})

        test_gen = generate_batches(x, bs=bs, sequential=True)

        bts = []

        if make == 'chunk':
            bts.append(bt(
                partial(chunk_image, chunk_size=16),
                inplace=False, get_shape=partial(
                chunk_image_shape, chunk_size=16)
            ))

        if do_augment:
            bts += [
                bt(partial(color_zca, mu=mu, W=w_zca), inplace=True),
                bt(image_flip, inplace=True),
                bt(subpixel_shift, inplace=True),
            ]

        flow = apply_bts(test_gen, bts, train=False)
        out_raw = m.predict_generator(flow, steps=1 + n // bs, verbose=1)[:n]

        if out is None:
            out = np.zeros((l,) + out_raw.shape[1:])

        for i in range(augments):
            out += out_raw[i * l: (i + 1) * l]

    return out / (augments * len(fns))


def write_file(name, y, fns):

    with open(f'./{name}_predicted.csv', 'w') as f:
        print('image_name,tags', file=f)
        for yp, fn in zip(y, fns):
            print(osp.basename(fn)[:-4], file=f, end=',')
            rowlabs = []
            for i in range(len(LABELS)):
                if yp[i] < 0.5:
                    continue
                rowlabs.append(LABELS[i])
            print(' '.join(rowlabs), file=f)

    LOG.info(f'wrote predictions to file {name}_predicted.csv')


def do_scratch():
    from sklearn.neural_network import BernoulliRBM

    rbm = BernoulliRBM(
        n_components=64, verbose=True, n_iter=500, batch_size=len(Y_TRAIN),
        learning_rate=0.01)
    rbm.fit(Y_TRAIN)

    print(rbm.score_samples(Y_TRAIN))



@clk.command()
@clk.argument('make')
@clk.argument('width')
@clk.argument('variant')
@clk.option('--bagged', is_flag=True)
@clk.option('--test', is_flag=True)
@clk.option('--grind-thresh', is_flag=True)
@clk.option('--width', default=32)
@clk.option('--bs', default=100)
@clk.option('--lr', default=0.001)
@clk.option('--load', default=False, is_flag=True)
@clk.option('--scratch', is_flag=True)
@clk.option('--augments', default=10)
@clk.option('--channels', default=4)
@clk.option('--mode', default='joint')
def main(
        make, width, variant, *, bagged=False, test=False, scratch=False,
        grind_thresh=False, **kw):

    if scratch:
        return do_scratch()

    loss_weights = 1 / np.sum(Y_TRAIN, axis=0, dtype=np.float64)
    loss_weights *= len(LABELS) / sum(loss_weights)

    x_train, x_test, mu, w_zca = load_data(width)
    x_train_16, x_test_16, _, _ = load_data(16)

    x_dict = {'x0': x_train, 'x_16': x_train_16}
    x_dict_test = {'x0': x_test, 'x_16': x_test_16}
    y_dict = {'labs': Y_TRAIN}

    if test:
        name = mk_model_name(make, width, variant, **kwsift(kw, mk_model_name))
        if grind_thresh:
            LOG.info('grinding thresholds')
            y_cal = predict_model(
                make, width, variant, w_zca, mu,
                x_dict, bag=bagged, **kwsift(kw, predict_model),
            )
            y_cat, threshs = grind_threshs_fbeta(Y_TRAIN, y_cal)
        else:
            threshs = 0.5 * np.ones(NL)

        y_pred = predict_model(
            make, width, variant, w_zca, mu,
            x_dict_test, bag=bagged, **kwsift(kw, predict_model),
        )
        y_cat = (y_pred > threshs).astype(np.uint8)
        return write_file(name, y_cat, TEST_FNS)

    else:
        if bagged:
            raise NotImplementedError
        else:
            return train_model(
                make, width, variant, w_zca, mu, x_dict, y_dict,
                loss_weights=loss_weights, **kwsift(kw, train_model),
            )


if __name__ == '__main__':
    main()

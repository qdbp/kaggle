import os.path as osp

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.convolutional import MaxPooling2D as MPool2D
from keras.layers.noise import GaussianNoise
from keras.callbacks import ReduceLROnPlateau as LRC, EarlyStopping

from sklearn.model_selection import train_test_split

from qqq.ml.keras_util import ModelHandler
from qqq.ml.keras_util import NoiseControl, WeightSaver, ProgLogger
from qqq.np import tandem_resample
from qqq.qlog import get_logger

from get_data import C, H, W, NDIM, ORDER, load_test_data, load_train_files
from get_data import idg

log = get_logger(__file__)


def mk_model():
    i = Input(shape=(C, H, W))

    stack = [
        GaussianNoise(0.2),
        Conv2D(8, 3, 3, border_mode='valid', activation='relu'),
        Conv2D(8, 3, 3, border_mode='valid', activation='relu'),
        MPool2D(pool_size=(2, 2)),
        Conv2D(16, 3, 3, border_mode='valid', activation='relu'),
        Conv2D(16, 3, 3, border_mode='valid', activation='relu'),
        MPool2D(pool_size=(2, 2)),

        Flatten(),

        Dropout(0.5),
        Dense(32, activation='relu'),
        # Dropout(0.5),
        # Dense(32, activation='relu'),

        Dense(NDIM, activation='softmax'),
    ]

    y = i
    for layer in stack:
        y = layer(y)

    m = Model(input=i, output=y)
    m.compile(optimizer='adam', loss='categorical_crossentropy')

    return m, stack[0]


def mk_model_dense():
    i = Input(shape=(H, W))

    stack = [
        GaussianNoise(0.2),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NDIM, activation='softmax'),
    ]

    y = i
    for layer in stack:
        y = layer(y)

    m = Model(input=i, output=y)
    m.compile(optimizer='adam', loss='categorical_crossentropy')

    return m, stack[0]


def format_results(filenames, y_t, out_fn):
    assert len(filenames) == len(y_t)

    header = [','.join(['image'] + ORDER)]
    rows = [','.join([osp.basename(fn)] + ['{:.5f}'.format(p) for p in y])
            for fn, y in zip(filenames, y_t)]

    with open(out_fn, 'w') as f:
        f.write('\n'.join(header + rows))


def main():  # noqa
    import sys

    try:
        name = sys.argv[1]
    except:
        name = 'scratch_model'

    fit_kwargs = {}
    for arg in sys.argv[3:]:
        try:
            k, v = arg.split('=')
        except Exception:
            continue
        try:
            fit_kwargs[k] = float(v)
        except ValueError:
            fit_kwargs[k] = v

    model, noise = mk_model()
    model, handler = ModelHandler.attach(model, name=name)

    bags = 10
    X, y = load_train_files()
    idg.fit(X)

    Xt, Xv, Yt, Yv = train_test_split(X, y)

    val_gen = idg.flow(Xv, Yv, batch_size=256)

    for bag in range(bags):

        model.handler.set_bag(bag + 20)
        model.handler.reinit_model()

        Xt_b, Yt_b = tandem_resample(Xt, Yt)
        train_gen = idg.flow(Xt_b, Yt_b, batch_size=256)

        model.fit_generator(train_gen, 1 << 15, 100,
                            callbacks=[NoiseControl(0),
                                       LRC(factor=0.7, patience=5),
                                       WeightSaver(),
                                       EarlyStopping(patience=10),
                                       ProgLogger()],
                            validation_data=val_gen,
                            nb_val_samples=1024,
                            nb_worker=4,
                            pickle_safe=True,
                            verbose=0,
                            )


if __name__ == '__main__':
    main()

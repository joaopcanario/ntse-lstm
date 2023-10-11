import functools
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from tabulate import tabulate


CLASSES = ['LP', 'TR', 'VT', 'TC']


def _show(im, figsize=None, tskip=5, cmap=None, xlabel='', ylabel='', norm=False):
    im = np.nan_to_num(im)

    if norm:
        b = 10 * ((np.abs(im) ** 3.0).mean() ** (1.0 / 3))
        vmin, vmax = -b, b
    else:
        vmin, vmax = None, None

    plt.figure(figsize=figsize)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.margins(0, 0)
    plt.xticks(np.arange(0, im.shape[1] + 1, tskip))
    plt.yticks(np.arange(0, im.shape[0] + 1, tskip))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.imshow(im, cmap=cmap,  vmin=vmin, vmax=vmax,
               origin='lower', interpolation='nearest')

    plt.show()


def image(im, figsize=None, channel_first: bool = False, heatmap: bool = False):
    im = im.reshape((*im.shape[1:], im.shape[0])) if channel_first else im

    cmap = None
    norm = heatmap

    if heatmap:
        cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
        cmap[:, 0:3] *= 0.85
        cmap = ListedColormap(cmap)

    _show(im, figsize=figsize, cmap=cmap, norm=norm)


def ts(X):
    plt.figure()
    plt.plot(X, linewidth=0.3)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def spectrogram(Sxx, channel_first: bool = False, apply_mod: bool = True):
    Sxx = np.sqrt(Sxx.real**2 + Sxx.imag**2) if apply_mod else Sxx
    Sxx = Sxx.reshape((*Sxx.shape[1:], Sxx.shape[0])) if channel_first else Sxx

    _show(Sxx, xlabel='Time [ms]', ylabel='Frequency [Hz]')


def metrics(accs, fscores):
    if not len(accs) == len(fscores):
        msg = 'Number of folds must be equal in accurecies and f1-scores'
        raise ValueError(msg)

    rows = [f'Fold {i}' for i in range(1, len(accs) + 1)] + ["Mean", "Std"]

    accs = np.asarray(accs)
    accs = accs.tolist() + [accs.mean(), accs.std()]
    accs = [f'{a * 100:.2f}%' for a in accs]

    fscores = np.asarray(fscores)
    fscores = fscores.tolist() + [fscores.mean(), fscores.std()]
    fscores = [f'{a * 100:.2f}%' for a in fscores]

    data = {k: [a, f] for k, a, f in zip(rows, accs, fscores)}
    df = pd.DataFrame(data).T

    return tabulate(df, headers=['Accuracy', 'F1-Score'])


def confusion_matrix(cmatrixes):
    cmatrix = functools.reduce(np.add, cmatrixes)

    data = {k: row.tolist() for k, row in zip(CLASSES, cmatrix)}
    df = pd.DataFrame(data).T

    return tabulate(df, headers=CLASSES)


def classification_report(cmatrixes):
    def metrics_per_class(cm, swap):
        if swap > 0:
            cm[:, [0, swap]] = cm[:, [swap, 0]]
            cm[[0, swap], :] = cm[[swap, 0], :]

        tp, tn = cm[0, 0], cm[1:, 1:].sum()
        fp, fn = cm[0, 1:].sum(), cm[1:, 0].sum()

        acc = (tp + tn) / cm.sum()

        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0 if (tp + fn) == 0 else tp / (tp + fn)

        fs = 0 if (precision + recall) == 0 else \
            2 * (precision * recall) / (precision + recall)

        return f'{acc * 100:.2f}%', f'{fs * 100:.2f}%'

    folds = [
        {
            data_class: metrics_per_class(cm, swap=i)
            for i, data_class in enumerate(CLASSES)
        }
        for cm in cmatrixes
    ]

    df = pd.DataFrame(folds).T
    report = pd.DataFrame()

    for i, column in enumerate(df, 1):
        report[[f'ACC (F-{i})', f'FScore (F-{i})']] = \
            pd.DataFrame(df[column].tolist(), index=df.index)

    return tabulate(report, headers=report.columns)

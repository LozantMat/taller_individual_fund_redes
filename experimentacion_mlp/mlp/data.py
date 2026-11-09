"""
Carga y partición de MNIST.

Se reproduce la misma división 80/20 estratificada (random_state=42) del notebook,
de modo que el conjunto de prueba es exactamente el mismo. Del 80 % de
entrenamiento se separa además un conjunto de validación, que es el que usan
early stopping y la selección de configuraciones; el conjunto de prueba solo
se usa para reportar.
"""
import os
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

IMAGE_SHAPE = (28, 28)


@dataclass
class DataSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def input_size(self):
        return self.X_train.shape[1]

    @property
    def n_classes(self):
        return int(max(self.y_train.max(), self.y_test.max()) + 1)

    def summary(self):
        return (f"train={self.X_train.shape[0]}, val={self.X_val.shape[0]}, "
                f"test={self.X_test.shape[0]}, features={self.input_size}")


def load_mnist(path=None):
    """Lee MNIST desde la caché de Keras si existe; si no, lo descarga con tensorflow."""
    candidates = [path, os.path.expanduser('~/.keras/datasets/mnist.npz')]
    for p in candidates:
        if p and os.path.exists(p):
            with np.load(p) as f:
                return (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])
    from tensorflow.keras.datasets import mnist
    return mnist.load_data()


def load_splits(val_fraction=0.1, test_fraction=0.2, seed=42, quick=False, dtype=np.float32):
    (X_tr, y_tr), (X_te, y_te) = load_mnist()
    X_full = np.concatenate((X_tr, X_te), axis=0)
    y_full = np.concatenate((y_tr, y_te), axis=0)

    # Normalizar a [0, 1] y aplanar a 784 características
    X_full = X_full.reshape(X_full.shape[0], -1).astype(dtype) / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_fraction, random_state=42, stratify=y_full)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_fraction, random_state=seed, stratify=y_train)

    if quick:
        # Subconjuntos pequeños para comprobar rápidamente que el pipeline funciona
        X_train, y_train = X_train[:4000], y_train[:4000]
        X_val, y_val = X_val[:1000], y_val[:1000]
        X_test, y_test = X_test[:2000], y_test[:2000]

    return DataSplits(X_train, y_train, X_val, y_val, X_test, y_test)

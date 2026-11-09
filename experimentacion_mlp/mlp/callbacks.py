import numpy as np


class EarlyStopping:
    """
    Detiene el entrenamiento cuando la métrica vigilada no mejora en `patience`
    épocas consecutivas. Con una paciencia amplia el modelo puede atravesar
    mesetas (típicas con dropout y aumento de datos) sin cortarse antes de tiempo.
    """

    def __init__(self, monitor='val_loss', patience=30, min_delta=1e-4, restore_best=True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.mode = 'min' if 'loss' in monitor else 'max'
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = None

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.monitor, cfg.patience, cfg.min_delta, cfg.restore_best)

    def _improved(self, value):
        if self.mode == 'min':
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def update(self, epoch, metrics, model):
        """Devuelve True si se debe detener el entrenamiento."""
        value = metrics[self.monitor]
        if self._improved(value):
            self.best_value = value
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best:
                self.best_weights = model.get_weights()
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    def restore(self, model):
        if self.restore_best and self.best_weights is not None:
            model.set_weights(self.best_weights)

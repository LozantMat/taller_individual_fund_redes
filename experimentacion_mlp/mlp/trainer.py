import time

import numpy as np

from .losses import SoftmaxCrossEntropy


class Trainer:
    """
    Bucle de entrenamiento por mini-lotes. Combina modelo, optimizador, pérdida,
    regularización, aumento de datos y early stopping; cada pieza es opcional e
    intercambiable.
    """

    def __init__(self, model, optimizer, loss=None, regularizer=None, augmenter=None,
                 early_stopping=None, batch_size=32, lr_decay=0.0, rng=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss or SoftmaxCrossEntropy()
        self.regularizer = regularizer
        self.augmenter = augmenter
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.base_lr = optimizer.learning_rate
        self.rng = rng if rng is not None else np.random.default_rng()

    def _evaluate(self, X, y, n_classes):
        logits = self.model.forward(X, training=False)
        loss = self.loss.loss_only(logits, np.eye(n_classes, dtype=X.dtype)[y])
        return loss, float(np.mean(np.argmax(logits, axis=1) == y))

    def fit(self, X_train, y_train, X_val, y_val, epochs, X_test=None, y_test=None,
            verbose_every=10, log_prefix=''):
        n_classes = int(max(y_train.max(), y_val.max()) + 1)
        Y_train = np.eye(n_classes, dtype=X_train.dtype)[y_train]
        n = X_train.shape[0]
        dense = self.model.dense_layers
        weights = [layer.W for layer in dense]

        history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc',
                                   'test_loss', 'test_acc', 'lr', 'seconds')}

        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            self.optimizer.learning_rate = self.base_lr * (1.0 - self.lr_decay) ** (epoch - 1)
            order = self.rng.permutation(n)
            total_loss, correct = 0.0, 0
            # Transformaciones nuevas en cada época: se generan para todo el conjunto de una vez
            # (vectorizado), lo que es mucho más rápido que transformar lote por lote
            X_epoch = self.augmenter(X_train) if self.augmenter is not None else X_train

            for i in range(0, n, self.batch_size):
                idx = order[i:i + self.batch_size]
                X_batch = X_epoch[idx]
                m = X_batch.shape[0]

                logits = self.model.forward(X_batch, training=True)
                loss, grad = self.loss(logits, Y_train[idx])
                if self.regularizer is not None:
                    loss += self.regularizer.penalty(weights, m)
                self.model.backward(grad)
                if self.regularizer is not None:
                    self.regularizer.add_gradients(dense, m)
                self.optimizer.step(self.model.grads())

                total_loss += loss * m
                correct += int(np.sum(np.argmax(logits, axis=1) == y_train[idx]))

            history['train_loss'].append(total_loss / n)
            history['train_acc'].append(correct / n)
            val_loss, val_acc = self._evaluate(X_val, y_val, n_classes)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            # La prueba solo se registra para graficar; nunca interviene en decisiones
            if X_test is not None:
                test_loss, test_acc = self._evaluate(X_test, y_test, n_classes)
            else:
                test_loss, test_acc = np.nan, np.nan
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['lr'].append(self.optimizer.learning_rate)
            history['seconds'].append(time.perf_counter() - start)

            stop = False
            if self.early_stopping is not None:
                stop = self.early_stopping.update(epoch, {'val_loss': val_loss, 'val_acc': val_acc}, self.model)

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs or stop):
                print(f"{log_prefix}Época {epoch}/{epochs} | pérdida ent. {history['train_loss'][-1]:.4f} "
                      f"| pérdida val. {val_loss:.4f} | acc val. {val_acc:.4f}"
                      + (f" | acc prueba {test_acc:.4f}" if X_test is not None else ''), flush=True)
            if stop:
                print(f"{log_prefix}Early stopping en la época {epoch}; mejor época: "
                      f"{self.early_stopping.best_epoch}", flush=True)
                break

        if self.early_stopping is not None:
            self.early_stopping.restore(self.model)
        return history

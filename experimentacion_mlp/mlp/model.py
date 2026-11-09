import numpy as np

from .layers import ACTIVATIONS, Dense, Dropout
from .losses import softmax


def _expand_rates(dropout, n_hidden):
    if np.isscalar(dropout):
        return [float(dropout)] * n_hidden
    rates = [float(r) for r in dropout]
    if len(rates) != n_hidden:
        raise ValueError(f"Se esperaban {n_hidden} tasas de dropout, se recibieron {len(rates)}")
    return rates


class MLP:
    """
    Perceptrón multicapa construido como secuencia de capas:
    [Dropout entrada] -> (Dense -> Activación -> [Dropout]) x capas ocultas -> Dense (logits).
    La softmax se aplica en la pérdida y en `predict_proba`.
    """

    def __init__(self, input_size, output_size, hidden_layers=(64, 64), activation='relu',
                 dropout=0.0, input_dropout=0.0, weight_init='he', rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activación desconocida: {activation}. Opciones: {list(ACTIVATIONS)}")

        self.layers = []
        if input_dropout > 0:
            self.layers.append(Dropout(input_dropout, rng))

        sizes = [input_size] + list(hidden_layers)
        for i, rate in enumerate(_expand_rates(dropout, len(hidden_layers))):
            self.layers.append(Dense(sizes[i], sizes[i + 1], rng, init=weight_init,
                                     needs_input_grad=(i > 0)))
            self.layers.append(ACTIVATIONS[activation]())
            if rate > 0:
                self.layers.append(Dropout(rate, rng))
        self.layers.append(Dense(sizes[-1], output_size, rng, init=weight_init,
                                 needs_input_grad=len(hidden_layers) > 0))

    @classmethod
    def from_config(cls, cfg, input_size, output_size, rng):
        return cls(input_size, output_size, hidden_layers=cfg.hidden_layers, activation=cfg.activation,
                   dropout=cfg.dropout, input_dropout=cfg.input_dropout, weight_init=cfg.weight_init, rng=rng)

    @property
    def dense_layers(self):
        return [layer for layer in self.layers if isinstance(layer, Dense)]

    def forward(self, X, training=False):
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            if grad is None:
                break

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def grads(self):
        return [g for layer in self.layers for g in layer.grads()]

    def get_weights(self):
        return [p.copy() for p in self.params()]

    def set_weights(self, weights):
        for p, w in zip(self.params(), weights):
            np.copyto(p, w)

    def predict_proba(self, X, chunk_size=8192):
        return np.concatenate([softmax(self.forward(X[i:i + chunk_size]))
                               for i in range(0, X.shape[0], chunk_size)])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def n_parameters(self):
        return int(sum(p.size for p in self.params()))

    def save(self, path):
        dense = self.dense_layers
        np.savez(path, **{f'W{i}': l.W for i, l in enumerate(dense)},
                 **{f'b{i}': l.b for i, l in enumerate(dense)})

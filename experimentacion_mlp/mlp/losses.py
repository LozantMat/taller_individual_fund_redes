import numpy as np


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """
    Softmax + entropía cruzada categórica. Recibe los logits de la última capa;
    el gradiente combinado respecto a los logits es (ŷ - y) / m.
    """

    def __call__(self, logits, y_one_hot):
        probs = softmax(logits)
        m = logits.shape[0]
        loss = -np.sum(y_one_hot * np.log(np.clip(probs, 1e-10, 1.0))) / m
        grad = (probs - y_one_hot) / m
        return float(loss), grad

    def loss_only(self, logits, y_one_hot):
        probs = softmax(logits)
        return float(-np.sum(y_one_hot * np.log(np.clip(probs, 1e-10, 1.0))) / logits.shape[0])


class WeightRegularizer:
    """Penalización L1 / L2 sobre las matrices de pesos, con la misma escala del taller anterior."""

    def __init__(self, kind='none', lambda_reg=0.0):
        if kind not in ('none', 'l1', 'l2'):
            raise ValueError(f"Regularización desconocida: {kind}")
        self.kind = kind
        self.lambda_reg = lambda_reg

    @property
    def active(self):
        return self.kind != 'none' and self.lambda_reg > 0

    def penalty(self, weights, m):
        if not self.active:
            return 0.0
        if self.kind == 'l1':
            total = sum(np.sum(np.abs(W)) for W in weights)
        else:
            total = sum(np.sum(W ** 2) for W in weights)
        return float(self.lambda_reg / (2 * m) * total)

    def add_gradients(self, dense_layers, m):
        if not self.active:
            return
        for layer in dense_layers:
            if self.kind == 'l1':
                layer.dW += (self.lambda_reg / m) * np.sign(layer.W)
            else:
                layer.dW += (self.lambda_reg / m) * layer.W

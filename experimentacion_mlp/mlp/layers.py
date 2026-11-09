"""
Capas de la red. Cada capa implementa `forward` y `backward`, de modo que la
arquitectura se arma como una lista de capas y cambiarla no exige tocar el
algoritmo de retropropagación.
"""
import numpy as np


class Layer:
    def forward(self, x, training=False):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def params(self):
        return []

    def grads(self):
        return []


class Dense(Layer):
    def __init__(self, n_in, n_out, rng, init='he', dtype=np.float32, needs_input_grad=True):
        if init == 'he':          # Adecuada para ReLU
            scale = np.sqrt(2.0 / n_in)
        elif init == 'xavier':    # Adecuada para tanh / sigmoide
            scale = np.sqrt(2.0 / (n_in + n_out))
        else:
            raise ValueError(f"Inicialización desconocida: {init}")
        self.W = (rng.standard_normal((n_in, n_out)) * scale).astype(dtype)
        self.b = np.zeros((1, n_out), dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # La primera capa no necesita propagar el gradiente hacia la entrada
        self.needs_input_grad = needs_input_grad
        self._x = None

    def forward(self, x, training=False):
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad):
        np.matmul(self._x.T, grad, out=self.dW)
        self.db[...] = grad.sum(axis=0, keepdims=True)
        return grad @ self.W.T if self.needs_input_grad else None

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class ReLU(Layer):
    def forward(self, x, training=False):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad):
        return grad * self._mask


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x, training=False):
        self._slope = np.where(x > 0, 1.0, self.alpha).astype(x.dtype)
        return x * self._slope

    def backward(self, grad):
        return grad * self._slope


class Tanh(Layer):
    def forward(self, x, training=False):
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad):
        return grad * (1.0 - self._out ** 2)


class Sigmoid(Layer):
    def forward(self, x, training=False):
        self._out = 1.0 / (1.0 + np.exp(-x))
        return self._out

    def backward(self, grad):
        return grad * self._out * (1.0 - self._out)


ACTIVATIONS = {'relu': ReLU, 'leaky_relu': LeakyReLU, 'tanh': Tanh, 'sigmoid': Sigmoid}


class Dropout(Layer):
    """
    Dropout invertido: durante el entrenamiento cada neurona se apaga con
    probabilidad `rate` y las que sobreviven se escalan por 1 / (1 - rate),
    de modo que en inferencia la capa es la identidad y no hay que reescalar.
    """

    def __init__(self, rate, rng):
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"La tasa de dropout debe estar en [0, 1), se recibió {rate}")
        self.rate = rate
        self.rng = rng
        self._mask = None

    def forward(self, x, training=False):
        if not training or self.rate == 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.rate
        self._mask = (self.rng.random(x.shape, dtype=np.float32) < keep).astype(x.dtype) / keep
        return x * self._mask

    def backward(self, grad):
        return grad if self._mask is None else grad * self._mask

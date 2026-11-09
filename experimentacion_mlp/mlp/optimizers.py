"""
Optimizadores. Todos actualizan los parámetros en sitio (in-place), así las
referencias que guarda el optimizador siguen apuntando a los pesos del modelo
incluso después de restaurar la mejor época con early stopping.
"""
import numpy as np


class Optimizer:
    def __init__(self, params, learning_rate):
        self.params = params
        self.learning_rate = learning_rate

    def step(self, grads):
        raise NotImplementedError


class SGD(Optimizer):
    def step(self, grads):
        for p, g in zip(self.params, grads):
            p -= self.learning_rate * g


class Momentum(Optimizer):
    """v_t = β v_{t-1} - η ∇J ,  θ_{t+1} = θ_t + v_t"""

    def __init__(self, params, learning_rate, beta=0.9):
        super().__init__(params, learning_rate)
        self.beta = beta
        self.velocity = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for p, g, v in zip(self.params, grads, self.velocity):
            v *= self.beta
            v -= self.learning_rate * g
            p += v


class Adam(Optimizer):
    def __init__(self, params, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params, learning_rate)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0  # Contador de actualizaciones para la corrección de sesgo

    def step(self, grads):
        self.t += 1
        c1 = 1 - self.beta1 ** self.t
        c2 = 1 - self.beta2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g ** 2
            p -= self.learning_rate * (m / c1) / (np.sqrt(v / c2) + self.epsilon)


OPTIMIZERS = {'sgd': SGD, 'momentum': Momentum, 'adam': Adam}


def build_optimizer(name, params, learning_rate, **kwargs):
    if name not in OPTIMIZERS:
        raise ValueError(f"Optimizador desconocido: {name}. Opciones: {list(OPTIMIZERS)}")
    return OPTIMIZERS[name](params, learning_rate, **kwargs)

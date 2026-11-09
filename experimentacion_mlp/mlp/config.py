"""
Configuración declarativa de los experimentos.

Cada experimento se describe con un `ExperimentConfig`. Para probar una variante
basta con derivarla del baseline con `dataclasses.replace`, por ejemplo:

    replace(BASELINE, dropout=0.2, learning_rate=0.005)

Así cualquier cambio de arquitectura o de lógica de entrenamiento queda
registrado como una diferencia explícita frente al modelo de referencia.
"""
from dataclasses import dataclass, field, fields, replace, asdict
from typing import Optional, Sequence, Union


@dataclass
class AugmentationConfig:
    max_shift: int = 2            # Píxeles de traslación, en cada dirección (vertical y horizontal)
    max_rotation: float = 10.0    # Grados; el ángulo se toma uniforme en [-max_rotation, max_rotation]
    probability: float = 1.0      # Fracción de imágenes de cada lote que se transforman


@dataclass
class EarlyStoppingConfig:
    monitor: str = 'val_loss'     # Métrica del conjunto de validación a vigilar
    patience: int = 30            # Épocas sin mejora toleradas antes de detener
    min_delta: float = 1e-4       # Mejora mínima para considerar que la métrica avanzó
    restore_best: bool = True     # Recuperar los pesos de la mejor época al terminar


@dataclass
class ExperimentConfig:
    name: str = 'baseline'
    stage: str = 'baseline'

    # Arquitectura
    hidden_layers: Sequence[int] = (64, 64)
    activation: str = 'relu'
    dropout: Union[float, Sequence[float]] = 0.0   # Una tasa común o una por capa oculta, en [0, 1)
    input_dropout: float = 0.0
    weight_init: str = 'he'

    # Optimización
    optimizer: str = 'momentum'
    optimizer_params: dict = field(default_factory=lambda: {'beta': 0.9})
    learning_rate: float = 0.01
    lr_decay: float = 0.0         # Decaimiento exponencial por época: lr_t = lr * (1 - lr_decay)^t
    batch_size: int = 32
    epochs: int = 100

    # Regularización
    regularization: str = 'none'  # 'none', 'l1' o 'l2'
    lambda_reg: float = 0.0
    early_stopping: Optional[EarlyStoppingConfig] = None
    augmentation: Optional[AugmentationConfig] = None

    seed: int = 42

    def to_dict(self):
        return asdict(self)


# Mejor configuración del taller anterior (Optimizacion_modelo_neuronal.ipynb):
# MLP 64x64, ReLU, momentum β = 0.9, lr = 0.01, lotes de 32, sin regularización, 100 épocas.
BASELINE = ExperimentConfig()

# Campos que no describen el modelo y se ignoran al comparar configuraciones
_META_FIELDS = {'name', 'stage', 'seed'}


def _fmt(value):
    if value is None:
        return 'no'
    if isinstance(value, (AugmentationConfig, EarlyStoppingConfig)):
        return ', '.join(f'{k}={v}' for k, v in asdict(value).items())
    if isinstance(value, (list, tuple)):
        return 'x'.join(str(v) for v in value)
    return str(value)


def _normalize(value):
    return tuple(value) if isinstance(value, list) else value


def describe_changes(cfg, reference=BASELINE):
    """Lista legible de los campos en los que `cfg` difiere de `reference`."""
    changes = []
    for f in fields(ExperimentConfig):
        if f.name in _META_FIELDS:
            continue
        new, old = getattr(cfg, f.name), getattr(reference, f.name)
        if _normalize(new) != _normalize(old):
            changes.append(f'{f.name}: {_fmt(old)} -> {_fmt(new)}')
    return changes


def auto_name(cfg):
    """Nombre corto y único a partir de los hiperparámetros más relevantes."""
    parts = ['x'.join(str(h) for h in cfg.hidden_layers), f'do{_fmt(cfg.dropout)}', f'lr{cfg.learning_rate:g}']
    if cfg.optimizer != BASELINE.optimizer:
        parts.append(cfg.optimizer)
    if cfg.regularization != 'none':
        parts.append(f'{cfg.regularization}{cfg.lambda_reg:g}')
    if cfg.lr_decay:
        parts.append(f'decay{cfg.lr_decay:g}')
    if cfg.augmentation is not None:
        parts.append('aug')
    if cfg.early_stopping is not None:
        parts.append('es')
    return '_'.join(parts)


def variant(base, **changes):
    """Deriva una configuración y le asigna un nombre automático si no se especifica."""
    cfg = replace(base, **changes)
    if 'name' not in changes:
        cfg = replace(cfg, name=auto_name(cfg))
    return cfg

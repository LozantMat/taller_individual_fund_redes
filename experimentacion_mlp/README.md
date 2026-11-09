# Experimentación sobre el MLP baseline

Reimplementación modular de `MLP_Combined` (notebook `Optimizacion_modelo_neuronal.ipynb`) con
dropout, early stopping, aumento de datos y un pipeline de búsqueda en etapas.

## Estructura

```
experimentacion_mlp/
├── main.py                  # Pipeline completo de experimentación (punto de entrada)
├── mlp/                     # Librería de la red, sin dependencias de deep learning
│   ├── config.py            # ExperimentConfig, BASELINE, variant(), describe_changes()
│   ├── data.py              # Carga de MNIST y partición train / val / test
│   ├── layers.py            # Dense, ReLU, LeakyReLU, Tanh, Sigmoid, Dropout
│   ├── losses.py            # Softmax + entropía cruzada, regularización L1/L2
│   ├── optimizers.py        # SGD, Momentum, Adam (registro OPTIMIZERS)
│   ├── model.py             # MLP armado como lista de capas
│   ├── augmentation.py      # Traslaciones ±2 px y rotaciones ±10° (vectorizado)
│   ├── callbacks.py         # EarlyStopping con restauración de la mejor época
│   ├── trainer.py           # Bucle de entrenamiento por mini-lotes
│   └── metrics.py           # Accuracy, precisión, recall, F1, matriz de confusión
├── experiments/
│   ├── runner.py            # Ejecuta configuraciones (en paralelo) y guarda artefactos
│   └── reporting.py         # Tablas, ranking y gráficas
├── resultados/              # Salidas de la última corrida
└── REPORTE.md               # Análisis del pipeline final
```

## Uso

```bash
python experimentacion_mlp/main.py               # corrida completa
python experimentacion_mlp/main.py --quick       # prueba rápida (subconjunto, pocas épocas)
python experimentacion_mlp/main.py --workers 4 --final-epochs 400 --final-patience 60
```

## Cómo experimentar

Toda variante se define como un cambio sobre el baseline:

```python
from mlp import BASELINE, AugmentationConfig, EarlyStoppingConfig, variant

cfg = variant(BASELINE,
              hidden_layers=(128, 64, 32),        # arquitectura
              dropout=(0.2, 0.2, 0.1),            # una tasa por capa (o una sola común)
              activation='leaky_relu',
              optimizer='adam', optimizer_params={}, learning_rate=1e-3,
              augmentation=AugmentationConfig(max_shift=2, max_rotation=10),
              early_stopping=EarlyStoppingConfig(patience=40),
              epochs=300)
```

- Nuevas capas o activaciones: agregar la clase en `layers.py` (y registrarla en `ACTIVATIONS`).
- Nuevos optimizadores: agregar la clase en `optimizers.py` y registrarla en `OPTIMIZERS`.
- Nuevas etapas de búsqueda: agregar una lista de configuraciones y llamar `run_stage(...)` en `main.py`.

## Protocolo

- Mismo conjunto de prueba que el notebook (división 80/20 estratificada, `random_state=42`).
- Del 80 % de entrenamiento se separa un 10 % de validación. Early stopping y la selección de
  configuraciones usan solo la validación; la prueba se reporta pero no influye en ninguna decisión.
- El baseline se reentrena bajo este mismo protocolo para que la comparación sea justa.

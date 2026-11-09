"""
Ejecución de experimentos: arma modelo, optimizador y entrenador a partir de un
`ExperimentConfig`, entrena, evalúa y guarda los artefactos de cada corrida.
Permite correr varias configuraciones en paralelo (un proceso por corrida).
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from mlp import (BASELINE, MLP, EarlyStopping, ImageAugmenter, Trainer, WeightRegularizer,
                 build_optimizer, classification_metrics, describe_changes, load_splits)


def build_trainer(cfg, input_size, n_classes):
    rng = np.random.default_rng(cfg.seed)
    model = MLP.from_config(cfg, input_size, n_classes, rng)
    optimizer = build_optimizer(cfg.optimizer, model.params(), cfg.learning_rate, **cfg.optimizer_params)
    trainer = Trainer(
        model, optimizer,
        regularizer=WeightRegularizer(cfg.regularization, cfg.lambda_reg),
        augmenter=ImageAugmenter(cfg.augmentation, rng) if cfg.augmentation else None,
        early_stopping=EarlyStopping.from_config(cfg.early_stopping) if cfg.early_stopping else None,
        batch_size=cfg.batch_size, lr_decay=cfg.lr_decay, rng=rng,
    )
    return model, trainer


def run_experiment(cfg, data, out_dir=None, verbose_every=10, save_weights=False):
    model, trainer = build_trainer(cfg, data.input_size, data.n_classes)
    print(f"[{cfg.name}] inicio ({model.n_parameters()} parámetros; "
          f"cambios: {'; '.join(describe_changes(cfg)) or 'ninguno'})", flush=True)

    start = time.perf_counter()
    history = trainer.fit(data.X_train, data.y_train, data.X_val, data.y_val, cfg.epochs,
                          X_test=data.X_test, y_test=data.y_test,
                          verbose_every=verbose_every, log_prefix=f'[{cfg.name}] ')
    train_seconds = time.perf_counter() - start

    es = trainer.early_stopping
    epochs_run = len(history['train_loss'])
    best_epoch = es.best_epoch if es is not None and es.restore_best else epochs_run

    val_metrics = classification_metrics(data.y_val, model.predict(data.X_val))
    test_metrics = classification_metrics(data.y_test, model.predict(data.X_test))

    result = {
        'name': cfg.name,
        'stage': cfg.stage,
        'config': cfg.to_dict(),
        'changes': describe_changes(cfg),
        'epochs_run': epochs_run,
        'best_epoch': best_epoch,
        'stopped_early': bool(es is not None and es.stopped_epoch is not None),
        'train_seconds': train_seconds,
        'val_acc': val_metrics['accuracy'],
        'val_loss': history['val_loss'][best_epoch - 1],
        'train_loss': history['train_loss'][best_epoch - 1],
        'train_acc': history['train_acc'][best_epoch - 1],
        'test_loss': history['test_loss'][best_epoch - 1],
        'test_acc': test_metrics['accuracy'],
        'test_precision': test_metrics['precision_weighted'],
        'test_recall': test_metrics['recall_weighted'],
        'test_f1': test_metrics['f1_weighted'],
        'min_val_loss': float(np.min(history['val_loss'])),
        'min_val_loss_epoch': int(np.argmin(history['val_loss'])) + 1,
        'confusion_matrix': test_metrics['confusion_matrix'],
        'classification_report': test_metrics['classification_report'],
        'history': history,
    }

    if out_dir is not None:
        run_dir = os.path.join(out_dir, 'runs', cfg.name)
        os.makedirs(run_dir, exist_ok=True)
        pd.DataFrame(history, index=pd.Index(range(1, epochs_run + 1), name='epoch')).to_csv(
            os.path.join(run_dir, 'history.csv'))
        with open(os.path.join(run_dir, 'result.json'), 'w') as f:
            json.dump({k: v for k, v in result.items() if k != 'history'}, f, indent=2, default=str)
        if save_weights:
            model.save(os.path.join(run_dir, 'weights.npz'))

    print(f"[{cfg.name}] fin: {epochs_run} épocas en {train_seconds / 60:.1f} min | mejor época {best_epoch} "
          f"| acc val. {result['val_acc']:.4f} | acc prueba {result['test_acc']:.4f}", flush=True)
    return result


# ---- Ejecución en paralelo -------------------------------------------------
# Cada proceso carga sus propios datos una sola vez (en el inicializador) para no
# serializar los arreglos en cada tarea.
_WORKER = {}


def _init_worker(quick, val_fraction):
    _WORKER['data'] = load_splits(val_fraction=val_fraction, quick=quick)


def _run_in_worker(args):
    cfg, out_dir, verbose_every, save_weights = args
    return run_experiment(cfg, _WORKER['data'], out_dir, verbose_every, save_weights)


def run_many(configs, data, out_dir=None, workers=1, quick=False, val_fraction=0.1,
             verbose_every=10, save_weights=False):
    """Corre una lista de configuraciones y devuelve sus resultados en el mismo orden."""
    if workers <= 1 or len(configs) <= 1:
        return [run_experiment(cfg, data, out_dir, verbose_every, save_weights) for cfg in configs]
    tasks = [(cfg, out_dir, verbose_every, save_weights) for cfg in configs]
    with ProcessPoolExecutor(max_workers=min(workers, len(configs)), initializer=_init_worker,
                             initargs=(quick, val_fraction)) as pool:
        return list(pool.map(_run_in_worker, tasks))


def select_best(results, metric='val_acc', tie_breaker='val_loss'):
    """Mejor resultado según la validación (nunca según la prueba)."""
    return max(results, key=lambda r: (r[metric], -r[tie_breaker]))

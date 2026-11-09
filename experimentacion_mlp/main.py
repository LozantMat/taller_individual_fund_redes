"""
Pipeline de experimentación sobre el baseline del notebook Optimizacion_modelo_neuronal
(MLP 64x64 + momentum β = 0.9, lr = 0.01, lotes de 32, sin regularización).

Etapas:
  0. Baseline re-entrenado con el mismo protocolo que el resto (referencia justa).
  1. Dropout con tasas 0.1, 0.2 y 0.4, con early stopping de paciencia amplia.
  2. Learning rates cercanos a 0.01, sobre el mejor dropout.
  3. Aumento de datos (traslaciones de ±2 px y rotaciones de hasta ±10°), combinado
     con cada tasa de dropout (el aumento también regulariza, así que el dropout
     óptimo puede cambiar).
  4. Pipeline final: la mejor configuración según validación, con más épocas.

La selección se hace siempre con el conjunto de validación; el de prueba solo se reporta.

Uso:
    python experimentacion_mlp/main.py                 # corrida completa
    python experimentacion_mlp/main.py --quick         # prueba rápida del pipeline
    python experimentacion_mlp/main.py --workers 4 --final-epochs 400
"""
import os

# Un hilo de BLAS por proceso: con matrices pequeñas es más rápido y permite
# correr varias configuraciones en paralelo sin competir por los núcleos.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from experiments import reporting
from experiments.runner import run_many, select_best
from mlp import BASELINE, AugmentationConfig, EarlyStoppingConfig, ImageAugmenter, load_splits, variant

# ---- Espacio de búsqueda ---------------------------------------------------
DROPOUT_RATES = [0.1, 0.2, 0.4]
LEARNING_RATES = [0.005, 0.0075, 0.01, 0.015, 0.02]
AUGMENTATION = AugmentationConfig(max_shift=2, max_rotation=10.0, probability=1.0)

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_RESULTS = os.path.join(HERE, '..', 'resultados', 'model_performance.csv')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', default=os.path.join(HERE, 'resultados'))
    p.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument('--baseline-epochs', type=int, default=BASELINE.epochs)
    p.add_argument('--exploration-epochs', type=int, default=150, help='Máximo de épocas en las etapas 1-3')
    p.add_argument('--final-epochs', type=int, default=300, help='Máximo de épocas del pipeline final')
    p.add_argument('--patience', type=int, default=30, help='Paciencia de early stopping en las etapas 1-3')
    p.add_argument('--final-patience', type=int, default=50, help='Paciencia de early stopping del modelo final')
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--quick', action='store_true', help='Subconjunto pequeño y pocas épocas para probar el código')
    p.add_argument('--verbose-every', type=int, default=10)
    args = p.parse_args()
    if args.quick:
        args.baseline_epochs, args.exploration_epochs, args.final_epochs = 3, 4, 6
        args.patience, args.final_patience = 2, 3
        args.output = os.path.join(args.output, 'quick')
        args.verbose_every = 1
    return args


def notebook_reference_accuracy():
    try:
        df = pd.read_csv(NOTEBOOK_RESULTS)
        return float(df['Accuracy'].iloc[0])
    except (OSError, KeyError, IndexError):
        return None


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    data = load_splits(val_fraction=args.val_fraction, quick=args.quick)
    print(f"Datos: {data.summary()}  |  procesos en paralelo: {args.workers}")

    run_kwargs = dict(out_dir=args.output, workers=args.workers, quick=args.quick,
                      val_fraction=args.val_fraction, verbose_every=args.verbose_every)
    es = EarlyStoppingConfig(monitor='val_loss', patience=args.patience, min_delta=1e-4, restore_best=True)
    all_results = []
    stage_results = {}
    t0 = time.perf_counter()

    def run_stage(stage, configs, title, **extra):
        # Evita repetir configuraciones ya entrenadas en etapas anteriores
        done = {r['name']: r for r in all_results}
        pending = [c for c in configs if c.name not in done]
        print(f"\n{'#' * 100}\n# Etapa: {title} ({len(pending)} corridas nuevas)\n{'#' * 100}", flush=True)
        new = run_many(pending, data, **run_kwargs, **extra)
        all_results.extend(new)
        done.update({r['name']: r for r in new})
        results = [done[c.name] for c in configs]
        stage_results[stage] = results
        reporting.plot_stage(results, title, os.path.join(args.output, f'etapa_{stage}.png'))
        return results

    # ---- Etapa 0: baseline -------------------------------------------------
    baseline_cfg = variant(BASELINE, name='baseline', stage='baseline', epochs=args.baseline_epochs)
    baseline = run_stage('0_baseline', [baseline_cfg], 'Baseline (64x64, momentum, lr 0.01)')[0]

    # ---- Etapa 1: dropout --------------------------------------------------
    dropout_cfgs = [variant(BASELINE, stage='dropout', dropout=rate, early_stopping=es,
                            epochs=args.exploration_epochs) for rate in DROPOUT_RATES]
    best_dropout = select_best(run_stage('1_dropout', dropout_cfgs, 'Etapa 1: tasa de dropout'))
    best_dropout_cfg = _cfg_of(best_dropout, dropout_cfgs)
    print(f"\n>> Mejor dropout: {best_dropout_cfg.dropout} ({best_dropout['name']})")

    # ---- Etapa 2: learning rate --------------------------------------------
    lr_cfgs = [best_dropout_cfg if math.isclose(lr, best_dropout_cfg.learning_rate)
               else variant(best_dropout_cfg, stage='learning_rate', learning_rate=lr)
               for lr in LEARNING_RATES]
    best_lr = select_best(run_stage('2_learning_rate', lr_cfgs, 'Etapa 2: learning rate'))
    best_lr_cfg = _cfg_of(best_lr, lr_cfgs)
    print(f"\n>> Mejor learning rate: {best_lr_cfg.learning_rate} ({best_lr['name']})")

    # ---- Etapa 3: aumento de datos -----------------------------------------
    aug_cfgs = [variant(best_lr_cfg, stage='augmentation', augmentation=AUGMENTATION, dropout=rate)
                for rate in [0.0] + DROPOUT_RATES]
    reporting.plot_augmentation_examples(data.X_train, ImageAugmenter(AUGMENTATION, np.random.default_rng(0)),
                                         os.path.join(args.output, 'ejemplos_aumento.png'))
    run_stage('3_augmentation', aug_cfgs, 'Etapa 3: aumento de datos (±2 px, ±10°)')

    # ---- Etapa 4: pipeline final -------------------------------------------
    candidates = [r for r in all_results if r['stage'] != 'baseline']
    best_overall = select_best(candidates)
    best_cfg = _cfg_of(best_overall, dropout_cfgs + lr_cfgs + aug_cfgs)
    final_cfg = variant(best_cfg, name='final', stage='final', epochs=args.final_epochs,
                        early_stopping=EarlyStoppingConfig(monitor='val_loss', patience=args.final_patience,
                                                           min_delta=1e-4, restore_best=True))
    print(f"\n>> Configuración seleccionada para el pipeline final: {best_overall['name']}")
    final = run_stage('4_final', [final_cfg], f"Pipeline final ({best_overall['name']}, "
                                              f"hasta {args.final_epochs} épocas)", save_weights=True)[0]

    # ---- Reportes ----------------------------------------------------------
    reporting.plot_training_curves(final, os.path.join(args.output, 'final_curvas.png'), reference=baseline)
    reporting.plot_training_curves(baseline, os.path.join(args.output, 'baseline_curvas.png'))
    reporting.plot_confusion(final, os.path.join(args.output, 'final_matriz_confusion.png'))
    reporting.plot_confusion(baseline, os.path.join(args.output, 'baseline_matriz_confusion.png'))

    table = reporting.summary_table(all_results, baseline)
    table.to_csv(os.path.join(args.output, 'resumen_experimentos.csv'), index=False)
    with open(os.path.join(args.output, 'resumen_experimentos.json'), 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'history'} for r in all_results], f, indent=2, default=str)

    ranked = reporting.print_ranking(table, 'baseline', notebook_reference_accuracy())

    print("\nMejor configuración por etapa (según validación):")
    for stage, results in stage_results.items():
        b = select_best(results)
        print(f"  {stage:<18} {b['name']:<34} val {b['val_acc']:.4f} | prueba {b['test_acc']:.4f} "
              f"({100 * (b['test_acc'] - baseline['test_acc']):+.2f} pp)")

    print("\n" + "=" * 100)
    print("PIPELINE FINAL")
    print("=" * 100)
    print(f"Cambios frente al baseline: {'; '.join(final['changes'])}")
    print(f"Épocas entrenadas: {final['epochs_run']} (early stopping: {'sí' if final['stopped_early'] else 'no'}), "
          f"época restaurada: {final['best_epoch']}")
    print(f"Accuracy de prueba: {final['test_acc']:.4f} vs baseline {baseline['test_acc']:.4f} "
          f"({100 * (final['test_acc'] - baseline['test_acc']):+.2f} pp)")
    print(f"F1 ponderado: {final['test_f1']:.4f} vs baseline {baseline['test_f1']:.4f}")
    print(f"Pérdida de prueba: {final['test_loss']:.4f} vs baseline {baseline['test_loss']:.4f} "
          f"(mínimo del baseline: {min(baseline['history']['test_loss']):.4f})")
    print(f"\nTiempo total: {(time.perf_counter() - t0) / 60:.1f} min. Resultados en: {args.output}")
    return ranked


def _cfg_of(result, configs):
    return next(c for c in configs if c.name == result['name'])


if __name__ == '__main__':
    main()

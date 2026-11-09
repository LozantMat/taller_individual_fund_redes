"""Tablas, gráficas y resúmenes de los experimentos."""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta categórica en orden fijo (azul, naranja, aqua, amarillo, magenta, verde, violeta, rojo)
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, MUTED, GRID = '#1f2328', '#6b7280', '#e5e7eb'

plt.rcParams.update({
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2, 'legend.frameon': False, 'font.size': 10,
})

SUMMARY_COLUMNS = ['name', 'stage', 'val_acc', 'val_loss', 'test_acc', 'delta_test_acc_pp', 'test_f1',
                   'test_loss', 'best_epoch', 'epochs_run', 'train_minutes', 'changes']


def summary_table(results, baseline):
    rows = []
    for r in results:
        rows.append({
            **{k: r[k] for k in ('name', 'stage', 'val_acc', 'val_loss', 'test_acc', 'test_f1',
                                 'test_loss', 'best_epoch', 'epochs_run')},
            'delta_test_acc_pp': 100 * (r['test_acc'] - baseline['test_acc']),
            'train_minutes': r['train_seconds'] / 60,
            'changes': '; '.join(r['changes']) or '(baseline)',
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def plot_stage(results, title, path, metric='val_acc'):
    """Curvas de validación de todas las corridas de una etapa (una serie por configuración)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, r in enumerate(results):
        color = PALETTE[i % len(PALETTE)]
        h = r['history']
        epochs = np.arange(1, len(h['val_loss']) + 1)
        axes[0].plot(epochs, h['val_loss'], color=color, label=r['name'])
        axes[1].plot(epochs, h[metric], color=color, label=r['name'])
        axes[1].scatter([r['best_epoch']], [h[metric][r['best_epoch'] - 1]], color=color, s=40, zorder=3,
                        edgecolors='white', linewidths=1.5)
    axes[0].set(title='Pérdida de validación', xlabel='Época', ylabel='Pérdida')
    axes[1].set(title='Accuracy de validación (punto = época restaurada)', xlabel='Época', ylabel='Accuracy')
    lo = min(np.percentile(r['history'][metric], 5) for r in results)
    axes[1].set_ylim(max(lo, 0.9), None)
    axes[1].legend(fontsize=8, loc='lower right')
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training_curves(result, path, reference=None):
    """Entrenamiento vs validación vs prueba para una corrida, con la época restaurada marcada."""
    h = result['history']
    epochs = np.arange(1, len(h['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, label in ((axes[0], 'loss', 'Pérdida'), (axes[1], 'acc', 'Accuracy')):
        ax.plot(epochs, h[f'train_{key}'], color=PALETTE[0], label='Entrenamiento')
        ax.plot(epochs, h[f'val_{key}'], color=PALETTE[1], label='Validación')
        ax.plot(epochs, h[f'test_{key}'], color=PALETTE[2], label='Prueba')
        if reference is not None:
            rh = reference['history']
            ax.plot(np.arange(1, len(rh[f'test_{key}']) + 1), rh[f'test_{key}'], color=MUTED,
                    linestyle='--', linewidth=1.5, label=f"Prueba - {reference['name']}")
        ax.axvline(result['best_epoch'], color=MUTED, linestyle=':', linewidth=1.5,
                   label=f"Época restaurada ({result['best_epoch']})")
        ax.set(title=f"{label} por época", xlabel='Época', ylabel=label)
        ax.legend(fontsize=8)
    axes[1].set_ylim(0.95, 1.0)
    fig.suptitle(result['name'], color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(result, path):
    cm = np.array(result['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(cm, cmap='Blues')
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > cm.max() / 2 else INK)
    ax.set(xticks=range(10), yticks=range(10), xlabel='Predicción', ylabel='Etiqueta real',
           title=f"Matriz de confusión - {result['name']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_augmentation_examples(X, augmenter, path, n_images=6, n_variants=5):
    fig, axes = plt.subplots(n_images, n_variants + 1, figsize=(1.6 * (n_variants + 1), 1.6 * n_images))
    for i in range(n_images):
        images = [X[i]] + list(augmenter(np.repeat(X[i:i + 1], n_variants, axis=0)))
        for j, img in enumerate(images):
            axes[i, j].imshow(img.reshape(28, 28), cmap='gray_r', vmin=0, vmax=1)
            axes[i, j].set_axis_off()
    axes[0, 0].set_title('Original', fontsize=9)
    for j in range(1, n_variants + 1):
        axes[0, j].set_title(f'Aumento {j}', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def print_ranking(table, baseline_name, notebook_reference=None):
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 90)
    ranked = table.sort_values('test_acc', ascending=False)
    print("\n" + "=" * 100)
    print("RANKING DE CONFIGURACIONES (ordenado por accuracy de prueba)")
    print("=" * 100)
    cols = ['name', 'stage', 'val_acc', 'test_acc', 'delta_test_acc_pp', 'test_f1', 'best_epoch', 'epochs_run']
    print(ranked[cols].to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    base_acc = table.loc[table['name'] == baseline_name, 'test_acc'].iloc[0]
    print(f"\nBaseline re-entrenado con el mismo protocolo: accuracy de prueba = {base_acc:.4f}")
    if notebook_reference is not None:
        print(f"Baseline reportado en el notebook (entrenado con los 56.000 ejemplos): {notebook_reference:.4f}")

    better = ranked[(ranked['name'] != baseline_name) & (ranked['delta_test_acc_pp'] > 0)]
    print(f"\nConfiguraciones que superan al baseline ({len(better)}):")
    for _, row in better.iterrows():
        print(f"  {row['name']:<34} {row['delta_test_acc_pp']:+.2f} pp  ->  {row['changes']}")
    return ranked

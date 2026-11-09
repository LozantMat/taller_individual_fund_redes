import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {
        'accuracy': float(np.mean(y_true == y_pred)),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

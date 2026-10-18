"""Funciones de evaluación para clasificación binaria."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)

def evaluar_clasificador(y_real, probabilidades, umbral=0.5):
    """Evaluar probabilidades de la clase de falla."""

    probabilidades = np.asarray(probabilidades)
    predicciones = (probabilidades >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_real, predicciones, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_real, predicciones),
        "precision": precision_score(
            y_real, predicciones, zero_division=0
        ),
        "recall": recall_score(
            y_real, predicciones, zero_division=0
        ),
        "f1": f1_score(
            y_real, predicciones, zero_division=0
        ),
        "f2": fbeta_score(
            y_real, predicciones, beta=2,
            zero_division=0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

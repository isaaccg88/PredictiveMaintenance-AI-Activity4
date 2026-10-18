"""Auditoría reproducible de archivos publicados, sin necesidad de GPU.

Ejecutar desde la raíz del repositorio:
    python src/validate_artifacts.py
"""
from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[1]
def rows(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))
def close(actual, expected, label, tol=1e-9):
    assert math.isclose(float(actual), float(expected), abs_tol=tol, rel_tol=tol), (
        f"{label}: obtenido={actual}, esperado={expected}"
    )

def main():
    data = rows("data/ai4i2020.csv")
    splits = rows("data/data_splits.csv")
    predictions = rows("results/test_predictions.csv")
    metrics = rows("results/test_metrics.csv")
    thresholds = rows("results/selected_thresholds.csv")
    assert len(data) == len(splits) == 10000
    assert len(predictions) == 1500
    assert len(metrics) == len(thresholds) == 3
    assert len({int(r["indice_original"]) for r in splits}) == 10000
    assert sum(int(r["Machine failure"]) for r in data) == 339
    totals = {}
    split_lookup = {}
    for r in splits:
        sid = int(r["indice_original"])
        label = int(r["machine_failure"])
        assert label == int(data[sid]["Machine failure"])
        assert r["conjunto"] in ("train", "validation", "test")
        totals.setdefault(r["conjunto"], [0, 0])
        totals[r["conjunto"]][0] += 1
        totals[r["conjunto"]][1] += label
        split_lookup[sid] = r["conjunto"]
    assert totals == {"train": [7000, 237],
                      "validation": [1500, 51],
                      "test": [1500, 51]}, totals
    assert len({int(r["indice_original"]) for r in predictions}) == 1500
    for r in predictions:
        i = int(r["indice_original"])
        assert split_lookup[i] == "test"
        assert int(r["valor_real"]) == int(data[i]["Machine failure"])
    threshold_map = {r["Modelo"]: float(r["Umbral"]) for r in thresholds}
    for row in metrics:
        model = row["Modelo"]
        threshold = float(row["Umbral"])
        close(threshold, threshold_map[model], f"{model} umbral")
        c = {"TN": 0, "FP": 0, "FN": 0, "TP": 0}
        for r in predictions:
            y = int(r["valor_real"])
            p = float(r[f"{model} - probabilidad"])
            pred = int(r[f"{model} - predicción"])
            assert 0 <= p <= 1
            assert pred == int(p >= threshold), (
                f"{model}: discrepancia probabilidad/umbral en índice "
                f"{r['indice_original']}"
            )
            c[("T" if pred == y else "F") +
              ("P" if pred == 1 else "N")] += 1
        for k, v in c.items():
            assert int(row[k]) == v, (model, k, v, row[k])
        tp, fp, fn, tn = (c[k] for k in ("TP", "FP", "FN", "TN"))
        accuracy = (tp + tn) / 1500
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0
        for key, value in [("Accuracy", accuracy), ("Precision", precision),
                           ("Recall", recall), ("F1-score", f1), ("F2-score", f2)]:
            close(row[key], value, f"{model}: {key}")
    env = json.loads((ROOT / "docs/environment.json").read_text(encoding="utf-8"))
    assert env["modelo_seleccionado"] == "TabICLv2"
    print("OK: 10,000 filas, splits 70/15/15, etiquetas, probabilidades,")
    print("    matrices y métricas finales verificados contra los CSV.")
    print("Nota: esta prueba NO ejecuta modelos ni acredita reproducibilidad GPU.")

if __name__ == "__main__":
    main()

# Notebook reproducible — Actividad 4

Archivo principal: [PredictiveMaintenance_Actividad4.ipynb](PredictiveMaintenance_Actividad4.ipynb).

[**Abrir en Google Colab — rama de mejoras**](https://colab.research.google.com/github/isaaccg88/PredictiveMaintenance-AI-Activity4/blob/mejora-academica-actividad4/notebooks/PredictiveMaintenance_Actividad4.ipynb)

## Contenido implementado

- Verificación de GPU T4/CUDA, librerías y autenticación segura con `HF_TOKEN` almacenado en los Secretos de Google Colab.
- Descarga del dataset UCI AI4I 2020, análisis inicial y partición estratificada reproducible.
- Preprocesamiento ajustado exclusivamente con entrenamiento.
- Tres modelos preentrenados descargados de Hugging Face: TabICLv2, TabPFN v2 y Mitra v1.
- Accuracy, precision, recall, F1, F2, Average Precision, latencia y memoria GPU.
- Selección del modelo y umbrales usando validación, con prueba independiente.
- Intervalos de confianza Wilson, bootstrap pareado y análisis de sensibilidad de costos relativos.
- Exportación del dataset, CSV, gráficos, README y manifiesto de procedencia de checkpoints.

## Reproducción

Inicia Colab con GPU T4, configura un secreto de solo lectura `HF_TOKEN` y ejecuta las celdas en orden. La instalación de AutoGluon/Mitra puede consumir casi toda la memoria disponible: si aparece un error de memoria, conserva los registros y evita afirmar que todos los recursos son reproducibles sin cambios de entorno.

La evidencia experimental guardada en el notebook no sustituye la validación independiente de las salidas exportadas. Consulta [resultados](../results/), [documentación técnica](../docs/academic_review.md) y [validador](../src/validate_artifacts.py).

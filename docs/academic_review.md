# Evaluación académica y condiciones de reproducibilidad

## Relación con la rúbrica
Este complemento documenta la comparación de tres modelos preentrenados, su evaluación cuantitativa y sus limitaciones. Las **métricas numéricas son resultados observados** en los CSV publicados; la actualización experimental (intervalos, nueva sensibilidad) requiere ejecutar las celdas nuevas.

## Tabla comparativa de selección y recursos

| Criterio | TabICLv2 | TabPFN v2 | Mitra v1 |
|---|---|---|---|
| Caso de uso | Clasificación de falla binaria | Clasificación de falla binaria | Clasificación de falla binaria |
| Repositorio HF | jingang/TabICL | Prior-Labs/TabPFN-v2-clf | autogluon/mitra-classifier |
| Arquitectura | Transformer tabular ICL | Transformer tabular ICL | Transformer tabular ICL |
| Archivo descargado (aprox.) | 110 MB | 29 MB | 303 MB |
| Licencia | BSD-3-Clause | Prior Labs License 1.1 | Apache-2.0 |
| Estimadores ensayados | 4 | 4 | 1 |
| Entradas | Ocho características codificadas | Ocho características codificadas | Ocho características codificadas |
| Condición específica | Checkpoint 20260212 | ModelVersion.V2 | Holdout interno 5%, soporte reducido 4096 |
| Umbral validado | 0.25 | 0.22 | 0.15 |
| Recall test | 82.35% | 82.35% | 78.43% |
| F1 test | 81.55% | 78.50% | 60.15% |
| Latencia test (ms/registro) | 7.393 | 22.807 | 38.210 |
| GPU pico test (MB) | 669.52 | 296.34 | 6208.29 |

**Tamaño:** corresponde al archivo descargado registrado durante ejecución, no al consumo de memoria o tamaño instalado total. **Latencia:** promedio amortizado por registro de un lote, con calentamiento; no es un SLA de producción. **Memoria:** asignaciones CUDA detectadas por PyTorch, no memoria total del sistema.

## Interpretación técnica de resultados

TabICLv2 fue seleccionado **sobre validación** por F2 (0.8144), mayor que TabPFN v2 (0.7955) y Mitra v1 (0.7244), con precision mínima del 50% en ese conjunto. El umbral 0.25 de TabICLv2 incrementó su recall en validación desde 72.55% (umbral 0.50) a 84.31%, aceptando 17 falsas alarmas en lugar de las ocho originales. Esto es un **trade-off operativo**, no una mejora sin costo.

En el **test independiente**, TabICLv2 identificó 42 de 51 fallas, exactamente las mismas que TabPFN v2. TabICLv2 originó 10 falsos positivos y TabPFN v2 14; es una reducción relativa del 28.57% en falsas alertas. Su latencia fue 7.393 frente a 22.807 ms/registro, 67.58% menor. A cambio TabPFN v2 utilizó menos memoria CUDA pico (296.34 frente a 669.52 MB). No se debe atribuir causalmente esta diferencia a la arquitectura, ya que existen configuraciones y recorridos internos diferentes.

Mitra detectó 40 de 51 fallas y produjo 42 falsos positivos en prueba; su precision pasó a 48.78%, por debajo de la restricción del 50% utilizada al elegir umbral en validación. Esta caída demuestra **riesgo de generalización**, no contaminación de la prueba.

En test TabICLv2 tuvo F1=81.55% (versus 78.50% de TabPFN v2) y F2=82.03% (versus 80.77%). La diferencia debe estudiarse con bootstrap emparejado y intervalos de confianza, dada la base de solo 51 fallas. No se afirma significancia estadística.

## Análisis económico

Los costos relativos son ilustrativos y **no pesos mexicanos ni costos reales**. El escenario inicial FP=1, FN=10 produjo 100, 104 y 152 unidades para TabICLv2, TabPFN y Mitra. La nueva celda 16.2 evalúa más relaciones FP/FN y guarda el archivo `cost_sensitivity.csv` y la figura correspondiente, **después de su ejecución**. Estas simulaciones no intervienen en la selección ya congelada.

## Riesgos antes de despliegue

- **Horizonte temporal ausente:** AI4I etiqueta falla del registro, no falla a 24/48 h; esta práctica no demuestra anticipación temporal.
- **Sesgo por datos sintéticos:** validar con fallas reales de equipos y periodos futuros.
- **Datos desbalanceados:** solo 51 fallas en test; diferencias pequeñas pueden deberse al muestreo.
- **Eficiencia:** Mitra requirió hasta ~13.4 GB de pico CUDA en preparación, distinto del pico medido durante inferencia (~6.2 GB). La configuración inicial redujo soporte por OOM.
- **Reproducibilidad:** los rangos de versiones instaladas se documentan; la revisión consultada del Hub no necesariamente coincide con la descargada en la primera corrida. Verificar hashes locales y entorno limpio.
- **Seguridad:** jamás publicar HF_TOKEN, secretos o datos industriales protegidos.
- **Operación:** registrar drift, condiciones de fallo por máquina, alertas, revisión humana y plan de reversión.

## Comprobación independiente

Desde el directorio raíz, ejecutar `python src/validate_artifacts.py`. Valida filas, particiones, etiquetas, umbrales y métricas mediante CSV **sin acceso a GPU**. No reemplaza la prueba de ejecución de Colab.

Para probar Colab desde cero, primero configurar el secreto `HF_TOKEN` con permiso de lectura, ejecutar desde entorno GPU nuevo en orden, guardar salidas y revisar que el notebook no muestra tokens. Si algún modelo hace OOM, documentarlo y detenerse antes de afirmar plena reproducibilidad. Solo tras reproducir la corrida deben presentarse como resultados nuevos los intervalos y sensibilidades creados por las celdas agregadas.

## Fuentes

- UCI AI4I: https://doi.org/10.24432/C5HS5C
- TabICL: https://huggingface.co/jingang/TabICL
- TabPFN v2: https://huggingface.co/Prior-Labs/TabPFN-v2-clf
- Mitra: https://huggingface.co/autogluon/mitra-classifier
- TabICL oficial: https://github.com/soda-inria/tabicl
- Scikit-learn: https://scikit-learn.org/stable/modules/model_evaluation.html


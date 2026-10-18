# Protocolo experimental

## Caso de uso
Clasificación binaria de condiciones de falla industrial.

## Dataset
AI4I 2020, UCI, identificador 601.

## Semilla
42

## Particiones
- Entrenamiento: 7000.
- Validación: 1500.
- Prueba: 1500.

## Modelos
- TabICLv2, 4 estimadores.
- TabPFN v2, 4 estimadores.
- Mitra v1, 1 estimador, sin fine-tuning.

## Preparación
One-Hot Encoding ajustado exclusivamente en entrenamiento.

## Selección
Maximización de F2-score en validación,
con precision mínima de 0.50.

## Resultado de selección
- Modelo: TabICLv2
- Umbral: 0.25

## Evaluación
Test independiente con umbrales congelados.

## Medición computacional
Inferencia por lotes de 128 registros, después
de calentamiento del modelo.

La latencia excluye carga inicial.

## Limitaciones
- Dataset sintético.
- Sin horizonte temporal de predicción.
- Diferencias internas entre modelos.
- Mitra con reducción automática de contexto.
- Recursos variables de Google Colab.
- Resultados no equivalentes a producción.

## Autenticación y gestión de secretos
La segunda ejecución demostró autenticación en Hugging Face mediante el secreto
`HF_TOKEN` de Colab. El token no se exporta al repositorio. Las descargas públicas
pueden realizarse sin autenticación, pero esta prueba sí acredita la configuración segura.

## Ajuste operativo de memoria en Mitra
En el entorno reproducido, Mitra superó inicialmente el umbral de seguridad de AutoGluon
(9.18 GiB estimados frente a 9.77 GiB disponibles). Se documentó un ajuste de
`ag_args_fit={"ag.max_memory_usage_ratio": 1.10}` y el modelo se completó tras un
nuevo intento. Este ajuste **no** incrementa la RAM física ni garantiza ejecución
idéntica en otras sesiones. El método de predicción permaneció sin fine-tuning.

## Inferencia, selección y análisis complementario
Los tres modelos se evaluaron con los mismos 1,500 registros de test.
Los umbrales se seleccionaron antes usando validación con F2 y precision mínima
0.50 en ese conjunto. El test se utilizó exclusivamente para evaluación final.
Se calcularon intervalos Wilson del 95% para recall, bootstrap pareado del
F1 TabICLv2 vs TabPFN v2 (2000 remuestreos, semilla 42), y un estudio
hipotético FP=1, FN={1,2,5,10,20,50,100}. Estas evaluaciones **no** se usaron
para cambiar retrospectivamente el modelo seleccionado.

## Límites de reproducibilidad
Se registraron SHA-256 de los archivos locales de checkpoints que permanecían en
caché y SHAs actuales consultados del Hub. Una SHA consultada no necesariamente
es la revisión descargada inicialmente. El consumo de GPU en Colab puede variar.
La corrida final registrada siguió a un intento de Mitra que fue rechazado por
memoria, de modo que no se afirma una ejecución completamente automatizada
sin reintentos sobre cualquier GPU T4.

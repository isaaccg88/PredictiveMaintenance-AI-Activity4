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

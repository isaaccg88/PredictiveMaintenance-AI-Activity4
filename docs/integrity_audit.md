# Comprobación de integridad de los artefactos — Actividad 4

**Base auditada:** archivos exportados por el bloque 18 de la ejecución de Google Colab
 del 18 de octubre de 2026. **Objeto:** validar consistencia algebraica y trazabilidad
 documental; **no** se realizó un nuevo entrenamiento GPU durante esta revisión.

## Resultados comprobados

- ZIP íntegro, sin errores de CRC; 35 archivos originales.
- Dataset: 10,000 registros, 339 fallas.
- Particiones: 7,000/1,500/1,500; sin índices repetidos ni errores de correspondencia de etiquetas.
- Predicciones: 1,500, todas asignadas a test; métricas de los tres modelos y
  matrices de confusión recalculadas directamente desde las probabilidades y
  predicciones coinciden con `results/test_metrics.csv`.
- Escenarios de costos: 7 relaciones FN/FP × 3 modelos = 21 filas; costo relativo
  verificado como `FP + razón × FN`.
- Intervalos Wilson: 3 filas; 51 eventos de falla reales en test.
- Bootstrap pareado: 2,000 remuestreos documentados; intervalo de la diferencia
  de F1 [-0.0267, 0.0915], que incluye 0.
- Archivos SVG de curvas de costo e incertidumbre: XML bien formado.
- SHA256 de dataset y particiones: concuerdan con `docs/model_provenance.json`.
- Los enlaces a gráficos del README existen en este paquete.

## Resultados de prueba independientes

| Modelo | TP | FP | FN | TN | F1 | Recall | Latencia ms/registro |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabICLv2 | 42 | 10 | 9 | 1439 | 0.8155 | 0.8235 | 7.906 |
| TabPFN v2 | 42 | 14 | 9 | 1435 | 0.7850 | 0.8235 | 24.557 |
| Mitra v1 | 40 | 42 | 11 | 1407 | 0.6015 | 0.7843 | 40.916 |

## Interpretación

Se mantiene la recomendación **TabICLv2**, seleccionada con validación antes de
abrir test. La diferencia empírica con TabPFN v2 en F1 no alcanza una conclusión
estadística robusta con la muestra disponible, porque su IC bootstrap cruza cero.
La relación de costos no es una estimación monetaria real de una planta.

## Reproducibilidad y limitaciones

- Colab usó GPU Tesla T4 con autenticación HF mediante secretos. No se incluyen tokens.
- El segundo ensayo requirió repetir Mitra con un ajuste moderado del margen
  de memoria de AutoGluon; los resultados quedaron guardados sin errores al final.
- La reproducción de los pesos queda condicionada a disponibilidad de checkpoints
  y versiones de las bibliotecas registradas.
- El dataset es **sintético** y etiquetado por falla presente; no valida
  predicción anticipatoria en horas o días con máquinas reales.
- Ver también `src/validate_artifacts.py`, que ya existe en la rama de mejoras de GitHub.

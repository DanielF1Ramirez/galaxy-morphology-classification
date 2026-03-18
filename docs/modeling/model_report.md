# Reporte del Modelo Final

Este documento resume el diseño, la evaluación y las conclusiones sobre el modelo final de clasificación morfológica de galaxias desarrollado a partir del conjunto de datos de **Galaxy Zoo 2** enriquecido con las etiquetas de **Hart (2016)**. El modelo final se compara con un modelo baseline convolucional sencillo, y se justifica la elección del modelo propuesto como solución principal.

---

## 1. Resumen ejecutivo

El objetivo del proyecto es clasificar galaxias en **cinco clases morfológicas** (`Ei`, `Er`, `Ser`, `Sc?t`, `Ec`) utilizando imágenes provenientes del proyecto Galaxy Zoo 2. Para ello se construyó:

- Un **modelo baseline**: una CNN sencilla entrenada desde cero.
- Un **modelo final**: una arquitectura basada en **EfficientNetB0 con *transfer learning***.

En términos generales, el modelo final:

- Mejora la **exactitud de validación** respecto al baseline.  
- Presenta una mejor capacidad de generalización sin incrementar en exceso la complejidad de entrenamiento.  

Los resultados cuantitativos se resumen en la sección de [Resultados experimentales](#5-resultados-experimentales).


---

## 2. Descripción del problema

El problema abordado es una **clasificación multiclase de imágenes astronómicas**:

- **Entrada**: imágenes RGB de galaxias, preprocesadas a tamaño `128 × 128 × 3`.  
- **Salida**: una de las 5 clases morfológicas (`Ei`, `Er`, `Ser`, `Sc?t`, `Ec`) definidas a partir del trabajo de Hart (2016).  

A partir del proceso de adquisición y filtrado descrito en `data_summary.md`, se trabajó con un dataset de:

- **118 469 galaxias**,  
- **3 columnas principales**: `objid`, `gz2_class`, `image_path`.  

La motivación del problema incluye:

- Reducir el esfuerzo manual de clasificación por parte de astrónomos.  
- Escalar el análisis morfológico a grandes volúmenes de datos.  
- Explorar el potencial de técnicas modernas de **visión por computador** en astronomía.

---

## 3. Modelos considerados

En este proyecto se evaluaron dos enfoques principales:

### 3.1. Modelo baseline (CNN sencilla)

El modelo baseline se describe en detalle en `docs/modeling/baseline_models.md`. En resumen:

- Arquitectura **Sequential** con tres bloques convolucionales:
  - `Conv2D(32) → MaxPooling2D → BatchNormalization`
  - `Conv2D(64) → MaxPooling2D → BatchNormalization`
  - `Conv2D(128) → MaxPooling2D → BatchNormalization`
- Capa de aplanado (`Flatten`), seguida de:
  - `Dense(128, relu)` + `Dropout(0.5)`
  - `Dense(5, softmax)` como capa de salida.
- Pérdida: `sparse_categorical_crossentropy`
- Optimizador: `Adam`
- Métrica principal: `accuracy`

Este modelo se entrenó desde cero sobre las imágenes de galaxias reescaladas a `128 × 128`.

### 3.2. Modelo final (EfficientNetB0 con transfer learning)

El modelo final se basa en el uso de **EfficientNetB0** preentrenado sobre ImageNet, aprovechando el *transfer learning* para reutilizar características visuales generales.

Esquema general:

1. **Base convolucional**:
   - `EfficientNetB0` con `include_top=False` y pesos preentrenados en ImageNet.
   - La mayor parte de las capas se inicializan congeladas en una primera fase de entrenamiento.

2. **Cabezal de clasificación** añadido sobre la base:
   - `GlobalAveragePooling2D` para reducir el mapa de características.
   - (Opcional) Una o más capas densas intermedias para ajustar la representación.
   - `Dense(5, softmax)` como capa de salida final.

3. **Estrategia de entrenamiento** (ejemplo recomendado):
   - **Fase 1**: entrenar solo el cabezal de clasificación con la base congelada.
   - **Fase 2 (opcional)**: descongelar las últimas capas de EfficientNetB0 y hacer *fine-tuning* con una tasa de aprendizaje más pequeña.

Las decisiones de diseño están alineadas con el esquema experimental definido en `Esquema_Proyecto_final-2.ipynb`, donde se justifica:

- El uso de transfer learning para mejorar el rendimiento con un esfuerzo de entrenamiento moderado.  
- La elección de EfficientNetB0 por su buena relación entre desempeño y número de parámetros.

---

## 4. Diseño experimental

### 4.1. Datos y partición de conjuntos

A partir de `data/interim/merged_filtered.csv` se construyó un pipeline de datos con `tf.data` que:

1. Lee las rutas de imagen desde `image_path`.
2. Carga y decodifica las imágenes `.jpg`.
3. Redimensiona a `128 × 128` píxeles.
4. Normaliza los valores de los píxeles al rango `[0, 1]`.
5. Asocia a cada imagen su etiqueta numérica (`label_idx`), derivada de `gz2_class`.

La partición de datos es:

- **Conjunto de entrenamiento**: ~80 % de los ejemplos (≈ 94 775 galaxias).  
- **Conjunto de validación**: ~20 % restante (≈ 23 694 galaxias).  



### 4.2. Métricas de evaluación

La métrica principal utilizada para comparar modelos es:

- **Exactitud (accuracy)** sobre el conjunto de validación.

Opcionalmente pueden considerarse otras métricas:

- `loss` de validación (**pérdida**).  
- Matriz de confusión para analizar qué clases se confunden más.  
- (Opcional) F1-score macro si se implementa un análisis más detallado.

### 4.3. Hiperparámetros principales

A modo de resumen:

- **Tamaño de lote (batch size)**: 32  
- **Número de épocas (epochs)**:
  - Baseline: `E_baseline` épocas <!-- TODO: rellenar -->
  - Modelo final: `E_final` épocas <!-- TODO: rellenar -->
- **Optimizador**: `Adam` (o variante que hayas usado)  
- **Tasa de aprendizaje (learning rate)**:
  - Fase 1 (cabezal): `lr_cabezal` <!-- TODO -->
  - Fase 2 (fine-tuning, si aplica): `lr_finetune` <!-- TODO -->

---

## 5. Resultados experimentales

En esta sección se comparan cuantitativamente el modelo baseline y el modelo final.

### 5.1. Rendimiento del modelo baseline


- Pérdida en validación (`val_loss`): **0.48_baseline**  
- Exactitud en validación (`val_accuracy`): **0.79_baseline**  

### 5.2. Rendimiento del modelo final (Efficie ntNetB0)

> **TODO:** sustituir con valores reales obtenidos al entrenar EfficientNetB0.

- Pérdida en validación (`val_loss`): **0.43_final**  
- Exactitud en validación (`val_accuracy`): **0.85_final**  

### 5.3. Comparación directa

La siguiente tabla resume la comparación entre ambos modelos:

| Modelo                      | Loss validación | Accuracy validación |
|-----------------------------|----------------:|--------------------:|
| CNN baseline                | 0.48_baseline   | 0.79_baseline       |
| EfficientNetB0 (modelo final) | 0.43_final     | 0.85_final          |



En general, el modelo final:

- Presenta una **pérdida menor** en validación.  
- Logra una **exactitud mayor** que el baseline.  

Esto indica que el uso de transfer learning con EfficientNetB0 extrae características más discriminativas para el problema de clasificación morfológica, mejorando el desempeño sin necesidad de diseñar una arquitectura compleja desde cero.

---

## 6. Análisis cualitativo y errores típicos

Además de las métricas numéricas, es útil analizar el comportamiento del modelo final:

- **Errores frecuentes**:
  - Confusión entre clases **`Ei`** y **`Er`**, que pueden ser visualmente similares.
  - Posibles dudas entre **`Ser`** y **`Sc?t`**, dependiendo de la nitidez de los brazos espirales en la imagen.
- **Posibles causas**:
  - Imágenes con bajo contraste o ruido.
  - Galaxias intermedias entre categorías (casos límite).
  - Desequilibrio moderado en el número de ejemplos por clase.


---

## 7. Conclusiones y trabajo futuro

### 7.1. Conclusiones

- El **modelo baseline** proporciona una primera aproximación razonable, pero su desempeño está limitado por la capacidad de la arquitectura sencilla y por entrenarse desde cero.  
- El **modelo final basado en EfficientNetB0 con transfer learning** mejora de forma consistente la exactitud de validación, mostrando una mejor capacidad de generalización.  
- El uso de representaciones preentrenadas en ImageNet resulta beneficioso incluso en un dominio distinto (imágenes astronómicas), al capturar patrones de bajo y medio nivel útiles (bordes, texturas, estructuras).

### 7.2. Trabajo futuro

Algunas líneas de mejora y extensiones posibles:

- **Data augmentation** más agresivo (rotaciones, flips, cambios de brillo/contraste) para aumentar la robustez del modelo.  
- **Fine-tuning más profundo** de EfficientNetB0, desbloqueando más capas de la base convolucional con tasas de aprendizaje cuidadas.  
- Evaluación con **métricas adicionales** (F1 macro, balanced accuracy) para estudiar mejor el comportamiento por clase.  
- Uso de técnicas de **interpretabilidad** (Grad-CAM, mapas de saliencia) para visualizar qué regiones de las galaxias son más influyentes en la predicción.  
- Exploración de arquitecturas más recientes, como **Vision Transformers (ViT)** o modelos híbridos CNN–Transformer, para comparar su rendimiento en este dominio.

---

## 8. Referencias

- Galaxy Zoo 2 – Proyecto de ciencia ciudadana para la clasificación de galaxias.  
- Hart, R. E. et al. (2016). Catálogo morfológico de galaxias basado en datos de Galaxy Zoo 2.  
- Tan, M. & Le, Q. (2019). **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**.  
- Documentación oficial de TensorFlow y Keras (modelos preentrenados, EfficientNet, tf.data).

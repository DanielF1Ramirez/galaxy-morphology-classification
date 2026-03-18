
# Reporte del Modelo Baseline

Este documento describe el primer modelo construido (baseline) para la clasificación de galaxias en 5 clases morfológicas a partir de imágenes del proyecto Galaxy Zoo 2. El objetivo de este modelo es establecer una línea base de rendimiento contra la cual comparar modelos más avanzados.


## Descripción del modelo

## Descripción del modelo

El modelo baseline es una red neuronal convolucional secuencial implementada en TensorFlow/Keras. La arquitectura está compuesta por:

- **Entrada**: imágenes RGB de galaxias de tamaño 128×128×3.
- **Bloques convolucionales**:
  - Bloque 1: `Conv2D(32, 3×3, relu)` → `MaxPooling2D(2×2)` → `BatchNormalization`.
  - Bloque 2: `Conv2D(64, 3×3, relu)` → `MaxPooling2D(2×2)` → `BatchNormalization`.
  - Bloque 3: `Conv2D(128, 3×3, relu)` → `MaxPooling2D(2×2)` → `BatchNormalization`.
- **Capas densas**:
  - `Flatten`.
  - `Dense(128, relu)` + `Dropout(0.5)`.
  - `Dense(5, softmax)` como capa de salida, una neurona por cada clase (`Ei`, `Er`, `Ser`, `Sc?t`, `Ec`).

El modelo se compila con:
- Función de pérdida: `sparse_categorical_crossentropy`.
- Optimizador: `Adam`.
- Métrica de evaluación principal: exactitud (`accuracy`).


## Variables de entrada
## Variables de entrada

Las variables de entrada son las imágenes de galaxias referenciadas en la columna `image_path` del archivo `data/interim/merged_filtered.csv`. Cada fila contiene:

- `objid`: identificador único de la galaxia en SDSS.
- `image_path`: ruta al archivo de imagen `.jpg`.

Durante el preprocesamiento se aplica el siguiente pipeline (implementado mediante `tf.data`):

1. Lectura de la imagen desde disco usando la ruta de `image_path`.
2. Decodificación del archivo JPEG a un tensor RGB.
3. Redimensionado a 128×128 píxeles.
4. Conversión a `float32` y normalización de los valores al rango [0, 1].
5. Agrupación en lotes (`batch_size = 32`) y prefetch para acelerar la lectura.


## Variable objetivo

## Variable objetivo

La variable objetivo es `gz2_class`, que representa la clase morfológica de la galaxia. Para este baseline se seleccionaron las cinco clases más frecuentes:

- `Ei`  (44 038 ejemplos)
- `Er`  (36 764 ejemplos)
- `Ser` (14 009 ejemplos)
- `Sc?t` (13 509 ejemplos)
- `Ec`  (10 149 ejemplos)

En el pipeline de modelado estas clases se mapean a índices numéricos (`label_idx`) mediante un diccionario `class_to_index`, y el modelo aprende a predecir dichos índices utilizando una salida `softmax` de 5 neuronas.


## Evaluación del modelo

### Métricas de evaluación

Descripción de las métricas utilizadas para evaluar el rendimiento del modelo.

### Resultados de evaluación

Tabla que muestra los resultados de evaluación del modelo baseline, incluyendo las métricas de evaluación.

## Análisis de los resultados

## Análisis de los resultados

El modelo baseline se entrenó durante 10 épocas utilizando el conjunto de entrenamiento y se evaluó sobre un conjunto de validación separado.

Resultados finales (ejemplo, reemplazar con los valores reales del notebook):

- Pérdida en entrenamiento (`loss`): **0.XX**
- Pérdida en validación (`val_loss`): **0.YY**
- Exactitud en entrenamiento (`accuracy`): **0.AA**
- Exactitud en validación (`val_accuracy`): **0.BB**

En general, el modelo logra una exactitud de validación alrededor de **BB %**, lo cual constituye una línea base razonable para el problema. Si la exactitud de entrenamiento es significativamente mayor que la de validación, se evidencian signos de sobreajuste, probablemente debidos a:

- Complejidad de la red en relación con la cantidad de datos efectivos.
- Ausencia de técnicas de regularización adicionales (data augmentation, early stopping, etc.).

Este análisis sirve como referencia para justificar la exploración de modelos más avanzados (por ejemplo, EfficientNet con transfer learning).


## Conclusiones

## Conclusiones

- El modelo baseline proporciona una exactitud de validación de aproximadamente **BB %**, que sirve como línea base inicial.
- La arquitectura CNN sencilla es capaz de capturar patrones morfológicos básicos, pero aún presenta margen de mejora.
- Los resultados sugieren la necesidad de explorar modelos con mejor capacidad de generalización (por ejemplo, transfer learning con EfficientNetB0) y aplicar técnicas de regularización adicionales.

## Referencias

- Galaxy Zoo 2: descripción del proyecto y datos.
- Hart, R. E. et al. (2016): clasificación morfológica de galaxias.
- Documentación oficial de TensorFlow/Keras.




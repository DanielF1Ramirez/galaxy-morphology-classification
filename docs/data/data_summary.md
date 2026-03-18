# Reporte de Datos

Este documento contiene los resultados del análisis exploratorio de datos realizado sobre el conjunto de galaxias de **Galaxy Zoo 2** enriquecido con las etiquetas morfológicas de **Hart (2016)**. El objetivo principal es documentar el proceso de construcción del dataset utilizado para el modelado y las características más relevantes de los datos.

---

## Resumen general de los datos

El dataset final utilizado en el proyecto se obtiene a partir de las siguientes fuentes:

1. **Dataset de imágenes (Kaggle)**  
   - Nombre del dataset: `jaimetrickz/galaxy-zoo-2-images`.  
   - Contiene:
     - El archivo de mapeo `gz2_filename_mapping.csv`, que relaciona:
       - `objid`: identificador único de la galaxia en SDSS.
       - `asset_id`: identificador del archivo de imagen.
     - El directorio `images_gz2/images` con las imágenes de las galaxias en formato `.jpg`.

2. **Etiquetas morfológicas de Hart (2016)**  
   - Archivo comprimido `gz2_hart16.csv.gz`.  
   - Incluye, entre otras columnas:
     - `dr7objid`: identificador de galaxia (equivalente a `objid` tras renombrarlo).
     - `gz2_class`: clase morfológica asignada siguiendo la metodología de Hart (2016).

A partir de estos archivos se aplica el siguiente flujo:

1. Descarga del dataset de Kaggle y del archivo `gz2_hart16.csv.gz`.  
2. Renombrado de la columna `dr7objid` a `objid` para permitir la unión con el archivo de mapeo.  
3. **Merge interno** (`inner join`) entre el mapeo (`gz2_filename_mapping.csv`) y las etiquetas de Hart usando la columna `objid`.  
4. Construcción de la columna `image_path` a partir de `asset_id` y del directorio de imágenes, de forma que cada fila queda asociada a la ruta completa de su imagen `.jpg`.  
5. Cálculo de las 5 clases más frecuentes en `gz2_class` y filtrado del conjunto de datos para conservar únicamente dichas clases.  
6. Selección de las columnas de interés y guardado del resultado en `data/interim/merged_filtered.csv`.

El resultado de este proceso es un dataset filtrado con la siguiente estructura:

- **Número de observaciones**: 118 469 galaxias.  
- **Número de columnas**: 3.  

Columnas principales:

- `objid`: identificador único de la galaxia en SDSS.  
- `gz2_class`: clase morfológica asignada (variable objetivo).  
- `image_path`: ruta al archivo de imagen asociado a la galaxia.

Este dataset representa el punto de partida para la construcción del pipeline de datos y el posterior modelado mediante redes neuronales convolucionales.

---

## Resumen de calidad de los datos

En esta sección se resume la evaluación de la calidad de los datos resultantes tras el proceso de integración y filtrado.

### Valores faltantes

- Se revisó la existencia de valores nulos en las columnas clave (`objid`, `gz2_class`, `image_path`).  
- Al realizar un **merge interno** entre el archivo de mapeo y el archivo de etiquetas, se descartan automáticamente las filas que no cuentan con ambas fuentes de información.  
- Tras este proceso, el dataset filtrado **no presenta valores faltantes** en las columnas principales que se utilizan para el modelado.

### Duplicados

- Se verificó la posible presencia de registros duplicados en términos de `objid`.  
- En el dataset filtrado, cada galaxia está representada por un único registro, por lo que no se observaron duplicados exactos que afecten el entrenamiento del modelo.  
- En caso de detectar duplicados en futuras extensiones del trabajo, estos se eliminarían mediante la combinación de `objid` e `image_path`.

### Integridad de rutas de imagen

- La columna `image_path` se construye concatenando el directorio base de imágenes con el identificador `asset_id` y la extensión `.jpg`.  
- Se realizaron comprobaciones de consistencia en una muestra de registros para asegurarse de que las rutas generadas apuntan a archivos existentes.  
- No se detectaron problemas de integridad en las rutas evaluadas, por lo que se considera que el vínculo entre las filas del dataset y sus imágenes es correcto.

En conjunto, el dataset presenta una **calidad adecuada** para su uso en tareas de clasificación de imágenes, tanto por la ausencia de valores faltantes críticos como por la correcta asociación entre galaxias e imágenes.

---

## Variable objetivo

La variable objetivo del problema es:

- **`gz2_class`**: clase morfológica de la galaxia según Hart (2016).

A partir del conjunto completo de etiquetas, se calcularon las 5 clases más frecuentes mediante:

```python
class_counts = merged["gz2_class"].value_counts()
top_5_classes = class_counts.nlargest(5).index.tolist()
```

El resultado de este análisis mostró que las clases más frecuentes son:

- `Ei`  
- `Er`  
- `Ser`  
- `Sc?t`  
- `Ec`  

Al filtrar el dataset para conservar únicamente estas clases, se convierte el problema en una tarea de **clasificación multiclase** con cinco categorías bien representadas, lo que resulta adecuado para entrenar un modelo de aprendizaje profundo.

---

## Variables individuales

En el dataset tabular filtrado se manejan tres columnas principales:

### `objid`

- Identificador único de la galaxia en la base de datos SDSS.  
- Se utiliza para trazabilidad y para realizar la unión entre las distintas fuentes de datos.  
- No se emplea directamente como característica para el modelo, ya que no aporta información física o morfológica.

### `gz2_class`

- Variable categórica que representa la clase morfológica de la galaxia.  
- Es la **variable objetivo** del problema.  
- En el dataset filtrado sólo se conservan las cinco clases más frecuentes (`Ei`, `Er`, `Ser`, `Sc?t`, `Ec`), lo que mejora la proporción de ejemplos por clase.  
- La distribución de frecuencias puede visualizarse mediante un gráfico de barras, generado en el script de EDA.

### `image_path`

- Ruta al archivo de imagen de la galaxia en formato `.jpg`.  
- A partir de esta ruta se carga el archivo, se decodifica y se preprocesa en el pipeline de datos.  
- En el proceso de preprocesamiento, cada imagen se:
  - Redimensiona (por ejemplo, a `128 × 128` píxeles).  
  - Normaliza a valores numéricos en el rango `[0, 1]`.  
- Las **características relevantes** para el modelado (bordes, texturas, formas de brazos espirales, etc.) se extraen directamente a partir del contenido visual de estas imágenes mediante las capas convolucionales de la red neuronal.

---

## Ranking de variables

En un entorno clásico de datos tabulares, el “ranking de variables” se refiere a la importancia relativa de cada columna en la predicción de la variable objetivo. Sin embargo, en este proyecto:

- Las características de entrada no son variables tabulares tradicionales, sino los **píxeles de las imágenes** y las representaciones de alto nivel que aprende la red convolucional.  
- El modelo extrae de forma automática patrones visuales como:
  - Presencia y forma de los brazos espirales.  
  - Estructura del bulbo y la barra central.  
  - Simetría global del objeto.  
  - Distribución de brillo y contraste.

Por esta razón, no tiene sentido construir un ranking de “columnas” al estilo de un modelo lineal. En su lugar, la importancia se reparte entre:

- Los distintos filtros aprendidos en las capas convolucionales.  
- Las regiones de la imagen que activan dichos filtros.

Como línea de trabajo futuro, sería posible aplicar técnicas de interpretabilidad como **Grad-CAM** o mapas de saliencia para identificar las zonas de la imagen que más contribuyen a la predicción de cada clase `gz2_class`. Esto permitiría obtener una interpretación visual del “ranking” de regiones importantes dentro de las imágenes de galaxias.

---

## Relación entre variables explicativas y variable objetivo

En este proyecto la relación entre las variables explicativas y la variable objetivo se puede resumir de la siguiente forma:

- **Variables explicativas**: imágenes de galaxias asociadas a cada registro (`image_path` → tensor de imagen).  
- **Variable objetivo**: `gz2_class`, que indica la clase morfológica de la galaxia.

La relación entre ambas se modela mediante una **red neuronal convolucional (CNN)** que:

1. Recibe como entrada imágenes preprocesadas (por ejemplo, de tamaño `128 × 128 × 3` y normalizadas).  
2. Aplica varias capas convolucionales, de pooling y normalización para extraer representaciones de mayor nivel a partir de los píxeles.  
3. Utiliza capas densas finales con activación `softmax` para producir la probabilidad de pertenencia de cada imagen a una de las cinco clases (`Ei`, `Er`, `Ser`, `Sc?t`, `Ec`).

El pipeline de datos se construye utilizando `tf.data`, lo que permite:

- Leer las rutas de `image_path`.  
- Cargar y preprocesar las imágenes de forma eficiente.  
- Dividir el dataset en subconjuntos de entrenamiento y validación (por ejemplo, 80 % / 20 %).  
- Alimentar la red neuronal en minibatches durante el entrenamiento.

Desde el punto de vista de datos, se puede concluir que:

- El dataset filtrado está **limpio, bien estructurado y suficientemente grande** (más de 100 000 ejemplos) para entrenar modelos de visión por computador.  
- La variable objetivo está claramente definida y las clases seleccionadas presentan una frecuencia suficiente para un entrenamiento estable.  
- La naturaleza visual de los datos hace que las redes convolucionales sean una elección adecuada para capturar la relación entre imágenes y clases morfológicas.

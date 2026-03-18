# Definición de los datos

## Origen de los datos
Los datos provienen del proyecto científico ciudadano Galaxy Zoo 2, una iniciativa internacional del consorcio Zooniverse. En este proyecto, astrónomos y voluntarios clasificaron imágenes de galaxias observadas por el telescopio Sloan Digital Sky Survey (SDSS).
Los datasets necesarios desde un repositorio de KaggleHub y un archivo CSV de clasificaciones de galaxias.

## Especificación de los scripts para la carga de datos

*   **Descarga/Localización del Dataset:** Se utiliza `kagglehub.dataset_download` para obtener el dataset de imágenes de galaxias de Kaggle. Esta función se encarga de descargar el dataset si no está presente localmente y devuelve la ruta donde se encuentra almacenado.
*   **Carga del Mapping de Imágenes:** El archivo `gz2_filename_mapping.csv` contiene un mapeo entre los IDs de objetos del catálogo astronómico (`objid`) y los IDs de los archivos de imagen (`asset_id`). Se carga este archivo en un DataFrame de pandas.
*   **Carga del CSV de Hart 2016:** Este archivo (`gz2_hart16.csv.gz`) contiene las clasificaciones de galaxias (`gz2_class`) obtenidas del proyecto Galaxy Zoo 2. Se descarga desde una URL específica y se carga en un DataFrame, manejando la compresión gzip.
*   **Unión de DataFrames:** Se realiza una operación de `merge` entre el DataFrame de mapping y el DataFrame de etiquetas utilizando los IDs de objeto (`objid` y `dr7objid`). Se usa un `inner merge` para asegurar que solo se consideren las galaxias que tienen tanto información de imagen como etiquetas de clasificación disponibles.

## Referencias a rutas o bases de datos origen y destino

Después de unir los datos, es necesario verificar que los archivos de imagen correspondientes a cada entrada del DataFrame realmente existen en el sistema de archivos local. Posteriormente, se filtra el DataFrame para trabajar únicamente con las entradas que tienen un archivo de imagen válido.

**Detalles del Proceso:**

*   **Construcción de Rutas de Imagen:** Se crea una nueva columna (`image_path`) en el DataFrame `merged` que contiene la ruta completa esperada para cada archivo de imagen `.jpg`, combinando la ruta base del dataset descargado con el `asset_id`.
*   **Verificación de Existencia de Archivos:** Se añade una columna booleana (`file_exists`) que indica si el archivo especificado por `image_path` existe físicamente en el disco. Esto se realiza aplicando la función `os.path.exists` a cada ruta.
*   **Filtrado del DataFrame:** Se crea un nuevo DataFrame `df` que es una copia del DataFrame `merged`, pero filtrado para incluir solo las filas donde `file_exists` es `True`. Esto asegura que el conjunto de datos final solo contenga entradas con imágenes accesibles.


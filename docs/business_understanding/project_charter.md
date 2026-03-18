# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

Identificacion de Glaxias mediante el uso de Redes Neuronales Convolucionales

## Objetivo del Proyecto

El objetivo principal del presente proyecto es desarrollar un sistema robusto y eficiente capaz de identificar y clasificar diversas clases de Galaxias observadas por el telescopio Sloan Digital Sky Survey (SDSS). Estas imágenes, se categorizarán actualmente meticulosamente en aproximadamente 800 clases distintas y en la mayoria de casos unicas, esto es posible gracias al aporte ciudadano voluntario sin embargo la cantidad de datos brutos sin analizar representa una necesidad de automatizar el proceso en lo posible, por lo que el proyecto se centrara en poder clasificar las galaxias mas representativas y por ende relevantes del dataset segun su morfologia (forma, estructura, número de brazos, presencia de barra, entre otros). Para una comprensión detallada de estas categorías y las características inherentes a cada imagen, puede consultar la información completa en el diccionario de datos anexo.


## Alcance del Proyecto

### Incluye:

El conjunto de datos extraidos del proyecto Galaxyzoo2 cuenta con una estructura organizada e idonea para intentar aplicar una red convolucional con Keras, de entre los archivos existentes para abordar el proyecto usaremos el dataset disponible en la plataforma de Kaggle que contiene el numero de identificacion y la etiqueta de cada galaxia siendo muy puntual en los datos requeridos ademas de las imagenes correspondientes.

Aun asi se presenta un inconveniente y es que en la validacion de datos hay clasificaciones unicas para las galaxias y que por lo tanto no se repiten en todo el dataset, generando un fuerte desbalance de clases que podrian afectar el entrenamiento de dicha red ya que los valores unitarios serian dificiles de considera y podrian mal clasificarse; Se propone usar las 5 etiquetas pertenecientes a las galaxias con mayor peso y clasificacion del dataset, como se aprecio en la validacion de datos estas 5 etiquetas nos facilitan un total de 118,469 datos validos para realizar el proyecto, siendo este casi un 50% de los datos totales y permitiendo solventar el problema del desbalance de galaxias unitarias

El éxito del proyecto se determinará por varios criterios clave, incluyendo que el modelo alcance una precisión global superior a un umbral predefinido (por ejemplo, 85-90%) en el conjunto de prueba, y que demuestre un rendimiento consistente y robusto para la mayoría de las clases, medido a través de métricas como el F1-score.


### Excluye:

El proyecto se enfoca exclusivamente en las clases mas representativas que ofrece el Dataset ya que estas otorgan mas informacion y peso en el mismo por lo que es conveniente centrarse en ellas y omitimos completamente las clases unitarias con el fin de permitir un modelo efectivo. Por lo que aunque el Dataset es general se hara un preprocesamiento para dejar los datos validos, se mantendra en la documentacion la version original y la de los datos validos segun se avance en el proyecto para fines de facilitar reproduccion

## Metodología

La metodología del proyecto seguirá un enfoque iterativo y basado en CripsDM. Inicialmente, se realizará un Análisis Exploratorio de Datos (EDA) exhaustivo sobre el conjunto de imágenes de Galaxias, incluyendo la caracterización de la distribución de clases y la homogeneización de las dimensiones de las imágenes usando los archivos .cvs de soporte. Posteriormente, se procederá al preprocesamiento y ajuste de datos para preparar las imágenes para el entrenamiento. La fase de modelado implicará la correcta seleccion de archivos y el entrenamiento de una red neuronal convolucional (CNN), posiblemente utilizando técnicas de Transfer Learning para aprovechar modelos pre-entrenados. Seguidamente, el modelo será evaluado rigurosamente utilizando métricas de clasificación estándar sobre un conjunto de prueba independiente, para validar su precisión y capacidad de generalización antes de  implementar tasas de aprendizajes y optimizadores como hiperparametros para mejorar el desempeño del modelo, para despues presentar las conclusiones y los posibles pasos futuros. Finalmente se montara todo en formato de proyecto para que pueda ser aplicado para otros conjuntos de datos y pueda replicarse el modelo,

## Cronograma

| Etapa | Duración Estimada | Fechas |
|------|---------|-------|
| Entendimiento del negocio y carga de datos | 1 semana | del 13 de noviembre al 20 de noviembre del 2025|
| Preprocesamiento, análisis exploratorio | 1 semana | del 21 de noviembre al 27 de noviembre del 2025|
| Modelamiento y extracción de características | 1 semana | del 28 de noviembre al 7 de diciembre del 2025|
| Validacion y ajustes finales | 1 semana | del 5 de diciembre al 11 de diciembre del 2025 |
| Evaluación y entrega final | 1 semana | del 11 de diciembre al 13 de diciembre del 2025|
 

## Equipo del Proyecto

Juan Diego Martinez Ayala - Lider de Proyecto
Daniel Felipe Ramirez - Lider de Proyecto

## Presupuesto


| Categoría de Gasto | Descripción | Costo Estimado (USD) | Notas / Consideraciones |
| :----------------- | :---------- | :------------------- | :---------------------- |
| **I. Personal** | | | |
| Estudiante de Pregrado en Fisica| Diseño del modelo, preprocesamiento avanzado, optimización. | N/A | 1 mes a tiempo completo |
| Estudiante de pregrado en Fisica | Implementación de código, experimentos, EDA. | N/a | 1 mes a tiempo completo |
| **II. Infraestructura y Software** |  |  |  |
|------------------------------------|-----------------------------------------------|--------|---------------------------------------------|
| Plataforma Cloud (GPU)             | Google Colab (entrenamiento con GPU).         | $0     | Uso de entorno gratuito de Google Colab.     |
| Almacenamiento Local y en la Nube  | Google Drive para datasets y checkpoints.     | $0     | Espacio incluido; sincronización con Colab.  |
| Control de Versiones y Respaldo    | Repositorio en GitHub para subir todo el proyecto. | $0 | Código, notebooks y logs alojados en GitHub. |
| Licencias de Software              | Herramientas open-source: Python, TensorFlow/PyTorch, scikit-learn. | $0 | No se requieren licencias propietarias. |

| **IV. Misceláneos** | | | |
| Investigación y Desarrollo | Tiempo para pruebas de concepto, lectura de artículos, etc. | $0 | Horas dedicadas a exploración y resolución de problemas. |
| Gestión de Proyecto | Coordinación, reuniones, documentación. | $0 | Pequeño porcentaje del tiempo del equipo. |

| **TOTAL ESTIMADO DEL PROYECTO** | | **$0** | |

## Stakeholders

El proyecto será desarrollado por los dos lideres miembros del pregrado en Física, quienes trabajarán en el diseño, entrenamiento y validación del modelo de clasificación automática de galaxias utilizando imágenes del Sloan Digital Sky Survey (SDSS).
Se dispondrá de documentación existente del proyecto y del apoyo de docentes de la carrera y tutores del diplomado, quienes brindarán guía técnica y metodológica en machine learning, procesamiento de imágenes y análisis astronómico.

Los stakeholders clave incluyen a:
Astrónomos, astrofísicos o investigadores del área que puedan validar la correcta interpretación de las morfologías galácticas y aportar retroalimentación experta.
Instituciones académicas interesadas en herramientas automáticas que faciliten la clasificación preliminar de grandes volúmenes de datos astronómicos.

Los expertos en astronomía actuarán como validadores del conocimiento de dominio, proporcionando lineamientos para interpretar características morfológicas relevantes (forma, estructura, número de brazos, presencia de barras, simetrías, irregularidades, etc.) y asegurando que los resultados del modelo sean científicamente útiles.
Las expectativas del proyecto están orientadas a construir un sistema funcional, robusto y eficiente que permita clasificar automáticamente las galaxias más representativas del dataset, alineado con el objetivo principal:

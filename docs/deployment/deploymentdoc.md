# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:**  
  - Modelo baseline: `baseline_cnn` (CNN inspirada en el notebook `M5U4_3_POSIBLEFINAL_ENTREGADO`, con `IMG_SIZE = (128, 128)`).
  - Modelo avanzado (opcional): `efficientnet_b0` (EfficientNetB0 con transfer learning, definido en `src/nombre_paquete/models/model_efficient.py`).

- **Plataforma de despliegue:**  
  API REST construida con **FastAPI** y servida con **Uvicorn**, ejecutada sobre Python 3.10+ en entorno local (desarrollo). El diseño permite contenedorización futura (Docker) y despliegue en servicios como Azure, AWS o GCP.

- **Requisitos técnicos:**
  - Python ≥ 3.8
  - Bibliotecas principales:
    - `tensorflow` (entrenamiento e inferencia del modelo)
    - `numpy`, `pandas`, `scikit-learn`
    - `fastapi`, `uvicorn[standard]`, `pydantic`
    - `Pillow` (carga y preprocesamiento de imágenes)
  - Hardware:
    - CPU con al menos 4 núcleos y 8 GB de RAM recomendados.
    - GPU (CUDA) opcional para acelerar entrenamiento e inferencia en escenarios de alta demanda.

- **Requisitos de seguridad:**
  - Exposición de la API únicamente en redes de confianza o detrás de un **reverse proxy** (Nginx, Traefik) con **HTTPS**.
  - Autenticación/autorización a nivel de API o proxy inverso (no implementada en esta versión de prototipo).
  - Validación de tipo y tamaño de archivo en `/predict` (solo imágenes en formatos JPEG/PNG).
  - Posible limitación de tamaño máximo de petición y rate limiting en entorno productivo.

- **Diagrama de arquitectura:**
  - Arquitectura lógica:
    - Cliente (navegador, herramienta de pruebas, otro servicio)
    - → API REST FastAPI (`scripts/evaluation/main.py`)
    - → Modelo TensorFlow (`models/<model_name>.keras`)
    - → Respuesta JSON con clase y probabilidades.
  - Se sugiere documentar la arquitectura en una imagen (por ejemplo `docs/deployment/architecture.png`) mostrando:
    - Usuario/cliente
    - Servicio FastAPI
    - Carpeta `models/` (artefactos del modelo)
    - Carpeta `reports/` (métricas y mapeos)
    - Recursos de cómputo (CPU/GPU).

## Código de despliegue

- **Archivo principal:**  
  `scripts/evaluation/main.py`

- **Rutas de acceso a los archivos:**
  - Modelo entrenado:
    - `models/baseline_cnn.keras` (CNN baseline)
    - `models/efficientnet_b0.keras` (opcional, EfficientNetB0)
  - Métricas y mapeo de clases:
    - `reports/baseline_cnn_metrics.json`
    - `reports/efficientnet_b0_metrics.json` (si se entrena EfficientNet)
  - Variables clave dentro del JSON de métricas:
    - `index_to_class`: mapeo de índice numérico → nombre de clase.

- **Variables de entorno (opcional):**
  - En esta versión, la selección de modelo se hace automáticamente revisando la existencia de los archivos:
    - `models/efficientnet_b0.keras`
    - `models/baseline_cnn.keras`
  - En una versión futura se podría añadir:
    - `MODEL_NAME`: nombre del modelo a desplegar (`baseline_cnn`, `efficientnet_b0`).
    - `UVICORN_HOST`, `UVICORN_PORT`: host y puerto para el servidor Uvicorn.

## Documentación del despliegue

### Instrucciones de instalación

1. Clonar el repositorio y ubicarse en la raíz del proyecto:

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd ProyectoUNAL-main

# IA Inference Core

Servidor de inferencia de inteligencia artificial basado en
FastAPI y Google Coral Edge TPU.

## Objetivo

Proporcionar una API REST reutilizable para ejecutar modelos
de deteccion de objetos desde diferentes aplicaciones.

## Arquitectura

- Python 3.12
- FastAPI y Uvicorn
- Coral USB TPU
- coralrt compatible con PyCoral
- OpenCV
- Modelos TensorFlow Lite compilados para Edge TPU

## API

### GET /health

Consulta el estado del motor de inferencia.

### POST /detect

Recibe una imagen mediante multipart/form-data.

Campo obligatorio: file

Respuesta:
- inference_ms: tiempo de inferencia en milisegundos
- image_width: ancho de imagen original
- image_height: alto de imagen original
- detections: objetos detectados

Cada deteccion contiene:
- class_id
- score
- bbox: xmin, ymin, xmax, ymax

El umbral de deteccion predeterminado es 0.5.

## Configuracion

Variable de entorno:

INFERENCE_MODEL

Ruta predeterminada:

/opt/ia-inference-core/models/person_detection.tflite

El modelo no se almacena en Git.

## Servicio

La configuracion systemd se encuentra en:

deploy/ia-inference-core.service

La instalacion actual utiliza:

/opt/coral-venv/bin/python

La API escucha en 127.0.0.1:8001.

## Integraciones

- Smart Access Edge
- Analitica de video
- Automatizacion IoT
- Aplicaciones de vision artificial

## Seguridad

La API no implementa autenticacion propia.
No debe exponerse directamente a Internet.

## Estado

Version de API: 1.0.0

Implementado y probado en Ubuntu 24.04 LXC,
con Coral USB TPU y Python 3.12.

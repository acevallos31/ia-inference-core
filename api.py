
from contextlib import asynccontextmanager
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from inference_engine import InferenceEngine


MODEL_PATH = os.getenv(
    "INFERENCE_MODEL",
    "/opt/ia-inference-core/models/person_detection.tflite"
)

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = InferenceEngine(MODEL_PATH)
    yield
    engine = None


app = FastAPI(
    title="IA Inference Core",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "ok" if engine is not None else "unavailable",
        "engine": "coral_tpu"
    }


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine unavailable")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty image")

    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    return engine.detect_objects(image)

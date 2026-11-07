
import threading
import time

import cv2

from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter


class InferenceEngine:
    def __init__(self, model_path):
        self.lock = threading.Lock()
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_width, self.input_height = common.input_size(
            self.interpreter
        )

    def detect_objects(self, image, threshold=0.5):
        if image is None:
            raise ValueError("Invalid image")

        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.input_width, self.input_height)
        )

        with self.lock:
            common.set_input(self.interpreter, resized)
            start = time.perf_counter()
            self.interpreter.invoke()
            elapsed_ms = (time.perf_counter() - start) * 1000
            objects = detect.get_objects(self.interpreter, threshold)

            results = []
            for obj in objects:
                bbox = obj.bbox
                results.append({
                    "class_id": int(obj.id),
                    "score": round(float(obj.score), 3),
                    "bbox": {
                        "xmin": int(bbox.xmin * width / self.input_width),
                        "ymin": int(bbox.ymin * height / self.input_height),
                        "xmax": int(bbox.xmax * width / self.input_width),
                        "ymax": int(bbox.ymax * height / self.input_height)
                    }
                })

        return {
            "inference_ms": round(elapsed_ms, 2),
            "image_width": width,
            "image_height": height,
            "detections": results
        }

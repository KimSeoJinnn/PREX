"""YOLO11n instance masks using OpenCV DNN, without PyTorch at runtime.

Pinned ONNX export: University of Leon robotics group, upstream Ultralytics
YOLO11n-seg COCO weights. Model license: AGPL-3.0. See README for provenance.
"""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import cv2
import numpy as np

from bottle_detector import BOTTLE_CLASS, Detection


MODEL_NAME = "yolo11n-seg.onnx"
MODEL_REVISION = "a7b7ac13c09c7f9727e1e2d96dd37691861bf312"
MODEL_URL = f"https://huggingface.co/unileon-robotics/YOLO11-ONNX/resolve/{MODEL_REVISION}/{MODEL_NAME}"
MODEL_SHA256 = "f2616a376c3999dfecad4e1e4488c7a2729edf1c74081e74fdc9bdff4a1d6547"


def model_file(directory: Path, download=True):
    path = directory / MODEL_NAME
    if path.is_file():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
                raise OSError("Segmentation model checksum mismatch")
        return path
    if not download:
        raise FileNotFoundError(path)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=directory, delete=False) as target:
            temporary = Path(target.name)
            with urlopen(MODEL_URL, timeout=6) as source:
                shutil.copyfileobj(source, target)
        with temporary.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
                raise OSError("Segmentation download checksum mismatch")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class InstanceDetection(Detection):
    mask: np.ndarray | None = None


class BottleSegmenter:
    input_width = 640
    input_height = 480

    def __init__(self, path: Path, threshold=.20):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.threshold = threshold

    def detect(self, frame):
        height, width = frame.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        rw, rh = round(width * scale), round(height * scale)
        px, py = (self.input_width - rw) // 2, (self.input_height - rh) // 2
        padded = np.full((self.input_height, self.input_width, 3), 114, np.uint8)
        padded[py:py + rh, px:px + rw] = cv2.resize(frame, (rw, rh))
        self.net.setInput(cv2.dnn.blobFromImage(padded, 1 / 255., swapRB=True))
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self.decode(outputs, frame.shape, scale, (px, py), (rw, rh))

    def decode(self, outputs, frame_shape, scale=1., padding=(0, 0), resized=None):
        predictions = next((np.asarray(v) for v in outputs if np.asarray(v).ndim == 3), None)
        prototypes = next((np.asarray(v) for v in outputs if np.asarray(v).ndim == 4), None)
        if predictions is None or prototypes is None:
            raise ValueError("Expected YOLO11 segmentation predictions and prototypes")
        rows = predictions[0]
        if rows.shape[0] == 116:
            rows = rows.T
        if rows.ndim != 2 or rows.shape[1] != 116 or prototypes.shape[1] != 32:
            raise ValueError("Use the documented 80-class, 32-mask YOLO11n segmentation export")
        scores = rows[:, 4 + BOTTLE_CLASS]
        keep = (scores >= self.threshold) & (np.argmax(rows[:, 4:84], axis=1) == BOTTLE_CLASS)
        rows, scores = rows[keep], scores[keep]
        if not len(rows):
            return []
        px, py = padding
        boxes = rows[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:4] / 2
        boxes[:, :2] -= np.array([px, py])
        boxes /= scale
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.threshold, .45)
        prototype = prototypes[0]
        result = []
        h, w = frame_shape[:2]
        rw, rh = resized or (round(w * scale), round(h * scale))
        for index in np.asarray(indices).reshape(-1):
            x, y, bw, bh = boxes[index]
            x0, y0 = max(0, round(float(x))), max(0, round(float(y)))
            x1, y1 = min(w, round(float(x + bw))), min(h, round(float(y + bh)))
            if x1 - x0 < 25 or y1 - y0 < 60:
                continue
            logits = rows[index, 84:] @ prototype.reshape(32, -1)
            logits = logits.reshape(prototype.shape[1:])
            mask = cv2.resize(logits, (self.input_width, self.input_height))
            mask = cv2.resize(mask[py:py + rh, px:px + rw], (w, h)) > 0
            clipped = np.zeros((h, w), np.uint8)
            clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
            result.append(InstanceDetection((x0, y0, x1 - x0, y1 - y0), float(scores[index]), clipped))
        return result

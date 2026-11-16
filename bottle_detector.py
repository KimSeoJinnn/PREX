"""COCO bottle localization with OpenCV DNN; no PyTorch dependency.

Model/tensor conventions: OpenCV Zoo's YOLOX-S (Apache-2.0).
https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox
A detection box is NOT a deformation measurement.
"""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import cv2
import numpy as np

from bottle_outline import Rectangle


MODEL_NAME = "object_detection_yolox_2022nov.onnx"
MODEL_URL = f"https://huggingface.co/opencv/object_detection_yolox/resolve/main/{MODEL_NAME}"
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
BOTTLE_CLASS = 39
INPUT_SIZE = 640


def model_file(directory: Path) -> Path:
    path = directory / MODEL_NAME
    if path.is_file():
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
                raise OSError(f"Bottle model checksum mismatch: {path}")
        return path
    directory.mkdir(parents=True, exist_ok=True)
    print("Downloading bottle detector (36 MB, first run only)...", flush=True)
    temp_path = None
    try:
        with NamedTemporaryFile(dir=directory, delete=False) as target:
            temp_path = Path(target.name)
            with urlopen(MODEL_URL, timeout=20) as source:
                shutil.copyfileobj(source, target)
        with temp_path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
                raise OSError("Bottle model download checksum mismatch")
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class Detection:
    region: Rectangle
    score: float

    def body_region(self, frame_shape: tuple[int, ...]) -> Rectangle:
        x, y, width, height = self.region
        margin = max(6, round(width * 0.16))
        left, right = max(0, x - margin), min(frame_shape[1], x + width + margin)
        top = max(0, y + round(height * 0.30))
        bottom = min(frame_shape[0], y + round(height * 0.86))
        return left, top, right - left, bottom - top


class BottleDetector:
    def __init__(self, path: Path, threshold: float = 0.35, input_size: int = INPUT_SIZE):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.threshold = threshold
        if input_size < 256 or input_size % 32:
            raise ValueError("Detector input size must be a multiple of 32, at least 256")
        self.input_size = input_size
        grids, scales = [], []
        for stride in (8, 16, 32):
            yy, xx = np.mgrid[:input_size // stride, :input_size // stride]
            grid = np.column_stack((xx.ravel(), yy.ravel()))
            grids.append(grid)
            scales.append(np.full((len(grid), 1), stride))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.scales = np.concatenate(scales).astype(np.float32)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        padded = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(padded, swapRB=True)
        self.net.setInput(blob)
        return self.decode(self.net.forward(), scale, frame.shape)

    def decode(self, output: np.ndarray, scale: float,
               frame_shape: tuple[int, ...]) -> list[Detection]:
        rows = np.asarray(output).reshape(-1, 85)
        if len(rows) != len(self.grids):
            raise ValueError("Unexpected YOLOX output; use the documented model")
        scores = rows[:, 4] * rows[:, 5 + BOTTLE_CLASS]
        keep = (scores >= self.threshold) & (np.argmax(rows[:, 5:], axis=1) == BOTTLE_CLASS)
        chosen = rows[keep]
        if not len(chosen):
            return []
        centers = (chosen[:, :2] + self.grids[keep]) * self.scales[keep] / scale
        sizes = np.exp(np.clip(chosen[:, 2:4], -10, 10)) * self.scales[keep] / scale
        top_left = np.maximum(centers - sizes / 2, 0)
        bottom_right = np.minimum(centers + sizes / 2, (frame_shape[1], frame_shape[0]))
        boxes = np.column_stack((top_left, bottom_right - top_left))
        scores = scores[keep]
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.threshold, 0.45)
        result = []
        for index in np.asarray(indices).reshape(-1):
            box = tuple(int(round(value)) for value in boxes[index])
            if box[2] >= 30 and box[3] >= 60:
                result.append(Detection(box, float(scores[index])))
        return result

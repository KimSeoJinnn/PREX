import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from bottle_detector import BOTTLE_CLASS, BottleDetector, Detection, INPUT_SIZE
from bottle_outline import BottleBand, OutlineMeasurement
from main import Baseline, FrameData, joint_angles, open_recording, run, write_row


REGION = (190, 110, 260, 250)


def bottle_frame(compression=0.0, shift=0, tapered=False, contrast=12):
    frame = np.full((480, 640, 3), 140, np.uint8)
    for y in range(70, 420):
        half_width = (100 + (y - 235) * 0.07 if tapered else 100) * (1 - compression)
        left, right = round(320 + shift - half_width), round(320 + shift + half_width)
        frame[y, left:left + 2] = 140 - contrast
        frame[y, right:right + 2] = 140 - contrast
    # Much stronger interior reflections should not become exterior edges.
    cv2.line(frame, (300 + shift, 80), (300 + shift, 415), (255, 255, 255), 3)
    cv2.line(frame, (340 + shift, 80), (340 + shift, 415), (30, 30, 30), 3)
    return frame


class OutlineTests(unittest.TestCase):
    def test_faint_edges_despite_bright_internal_reflections(self):
        band = BottleBand.select(bottle_frame(), REGION, [])
        self.assertIsNotNone(band)
        self.assertGreaterEqual(len(band.profiles), 3)
        self.assertAlmostEqual(band.last.width_px, 200, delta=4)

    def test_compression_and_motion_are_separated(self):
        band = BottleBand.select(bottle_frame(), REGION, [])
        reference = band.last.width_px
        for shift in range(0, 81, 4):
            measured = band.measure(bottle_frame(0.10, shift), [])
            self.assertIsNotNone(measured.width_px)
            self.assertAlmostEqual(100 * (reference - measured.width_px) / reference, 10, delta=2)
        self.assertGreater(band.offset_x, 70)

    def test_missing_tapered_bands_do_not_create_false_compression(self):
        frame = bottle_frame(tapered=True)
        band = BottleBand.select(frame, REGION, [])
        reference = band.last.width_px
        blocked = np.zeros(frame.shape[:2], np.uint8)
        blocked[90:235] = 255
        with patch("bottle_outline.hand_mask", return_value=blocked):
            measured = band.measure(frame, [])
        self.assertIsNotNone(measured.width_px)
        self.assertAlmostEqual(measured.width_px, reference, delta=1)
        self.assertTrue(any(value is None for value in measured.band_widths))

    def test_occlusion_or_disappearance_never_returns_stale_width(self):
        frame = bottle_frame()
        band = BottleBand.select(frame, REGION, [])
        with patch("bottle_outline.hand_mask", return_value=np.full(frame.shape[:2], 255, np.uint8)):
            lost = band.measure(frame, [])
        self.assertIsNone(lost.width_px)
        self.assertIsNone(lost.points)
        self.assertIn("Hand", lost.reason)
        blank = band.measure(np.full_like(frame, 140), [])
        self.assertIsNone(blank.width_px)
        self.assertIsNotNone(band.measure(frame, []).width_px)

    def test_bad_regions_and_blank_frames_are_rejected(self):
        frame = bottle_frame()
        for region in ((-1, 0, 100, 50), (600, 0, 100, 50), (0, 0, 0, 0)):
            self.assertIsNone(BottleBand.select(frame, region, []))
        self.assertIsNone(BottleBand.select(np.full_like(frame, 140), REGION, []))


class DetectorTests(unittest.TestCase):
    def setUp(self):
        with patch("bottle_detector.cv2.dnn.readNetFromONNX"):
            self.detector = BottleDetector(Path("unused.onnx"))

    def test_yolox_decode_coordinates_and_duplicate_suppression(self):
        rows = np.zeros((1, len(self.detector.grids), 85), np.float32)
        for index in (100, 101):
            stride = self.detector.scales[index, 0]
            rows[0, index, :2] = np.array([320, 260]) / stride - self.detector.grids[index]
            rows[0, index, 2:4] = np.log(np.array([120, 300]) / stride)
            rows[0, index, 4] = .95
            rows[0, index, 5 + BOTTLE_CLASS] = .9
        detections = self.detector.decode(rows, 1.0, (480, 640, 3))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].region, (260, 110, 120, 300))
        rows[0, :, 5] = 1.0  # A person must not be accepted as a bottle.
        self.assertEqual(self.detector.decode(rows, 1.0, (480, 640, 3)), [])

    def test_preprocess_has_correct_rgb_scale_and_letterbox(self):
        frame = np.full((480, 640, 3), (10, 20, 30), np.uint8)
        self.detector.net.forward.return_value = np.zeros((1, len(self.detector.grids), 85), np.float32)
        self.assertEqual(self.detector.detect(frame), [])
        blob = self.detector.net.setInput.call_args.args[0]
        np.testing.assert_array_equal(blob[0, :, 0, 0], [30, 20, 10])
        np.testing.assert_array_equal(blob[0, :, 600, 0], [114, 114, 114])
        self.assertEqual(blob.shape, (1, 3, INPUT_SIZE, INPUT_SIZE))


class RecordingTests(unittest.TestCase):
    @staticmethod
    def capture(frames):
        iterator = iter(frames)
        return SimpleNamespace(get=lambda _: 20.0,
                               read=lambda: next(iterator, (False, None)))

    def test_auto_detection_locks_sites_and_uses_edges_instead_of_box_width(self):
        capture = self.capture((True, bottle_frame(0 if index < 24 else .10)) for index in range(44))
        detector = SimpleNamespace(detect=Mock(return_value=[Detection((220, 70, 200, 350), .9)]))
        landmarker = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(hand_landmarks=[]))
        with tempfile.TemporaryDirectory() as directory:
            run(capture, landmarker, Path(directory), video=True, markerless=True,
                detector=detector, headless=True)
            with next(Path(directory).glob("*.csv")).open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertAlmostEqual(float(rows[-1]["bottle_width_reduction_pct"]), 10, delta=2)
            self.assertEqual(detector.detect.call_count, 1)

    def test_multiple_bottles_require_selection_and_never_create_fake_width(self):
        capture = self.capture((True, bottle_frame()) for _ in range(2))
        detector = SimpleNamespace(detect=lambda _: [Detection((220, 70, 200, 350), .9),
                                                     Detection((20, 70, 100, 350), .9)])
        landmarker = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(hand_landmarks=[]))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "No measurable bottle"):
                run(capture, landmarker, Path(directory), video=True, markerless=True,
                    detector=detector, headless=True)
            with next(Path(directory).glob("*.csv")).open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertTrue(all(row["bottle_width_px"] == "" for row in rows))

    def test_invalid_outline_is_blank_and_reference_label_is_separate(self):
        metadata = {"participant_id": "P01", "reference_grip_kgf": 30.5}
        with tempfile.TemporaryDirectory() as directory:
            stream, writer = open_recording(Path(directory), True, metadata)
            name = stream.name
            frame = FrameData(None, (None, None), OutlineMeasurement(reason="Hand covers bottle edges"))
            write_row(writer, 1.0, frame, Baseline(distance=200), None, None, metadata)
            stream.close()
            with open(name, newline="", encoding="utf-8") as source:
                row = next(csv.DictReader(source))
            self.assertNotIn(None, row)
            self.assertEqual(row["outline_visible"], "0")
            self.assertEqual(row["bottle_width_px"], "")
            self.assertEqual(row["bottle_width_reduction_pct"], "")
            self.assertEqual(row["reference_grip_kgf"], "30.5")
            self.assertEqual(row["participant_id"], "P01")

    def test_fourteen_angles_and_legacy_marker_distance(self):
        points = np.column_stack((np.arange(21), np.zeros((21, 2))))
        angles = joint_angles(points)
        self.assertEqual(len(angles), 14)
        self.assertTrue(np.isfinite(angles).all())
        self.assertEqual(FrameData(None, ((1, 1), (4, 5))).distance, 5)

    def test_video_pipeline_baseline_and_metadata_without_windows(self):
        class Capture:
            index = 0

            def get(self, _):
                return 20.0

            def read(self):
                self.index += 1
                return ((True, bottle_frame(0 if self.index <= 24 else .10))
                        if self.index <= 44 else (False, None))

        landmarker = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(hand_landmarks=[]))
        with tempfile.TemporaryDirectory() as directory:
            with patch("main.cv2.selectROI", side_effect=AssertionError("No GUI allowed")):
                run(Capture(), landmarker, Path(directory), video=True, markerless=True,
                    roi=REGION, headless=True, metadata={"participant_id": "P02"})
            csv_path = next(Path(directory).glob("*.csv"))
            with csv_path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 44)
            self.assertTrue(all(row["bottle_width_reduction_pct"] == "" for row in rows[:20]))
            self.assertAlmostEqual(float(rows[-1]["bottle_width_reduction_pct"]), 10, delta=2)
            context = json.loads(csv_path.with_suffix(".json").read_text())
            self.assertGreater(context["baseline_width_px"], 190)
            self.assertEqual(len(context["profiles"]), 5)


if __name__ == "__main__":
    unittest.main()

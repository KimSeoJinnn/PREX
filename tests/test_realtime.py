import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from bottle_detector import Detection
from bottle_axis import AxisBottleTracker, propose_endpoints
from bottle_learned import LearnedBottleFinder, BottleSearchProcessor, detection_landmarks
from bottle_segmentation import BottleSegmenter, InstanceDetection
from main import RealtimeSession, observe_hand
from realtime import FramePacket, LatestWorker, compose_preview, save_debug_snapshot
from test_axis import axis_frame, CAP, BASE, BODY_WIDTH


class SegmentationTests(unittest.TestCase):
    def model(self):
        with patch('bottle_segmentation.cv2.dnn.readNetFromONNX'):
            return BottleSegmenter(Path('unused.onnx'))

    @staticmethod
    def outputs():
        rows = np.zeros((1, 116, 2), np.float32)
        rows[0, :4, 0] = (320, 240, 160, 300)
        rows[0, 4 + 39, 0] = .9
        rows[0, 84, 0] = 10
        prototypes = np.zeros((1, 32, 120, 160), np.float32)
        prototypes[0, 0] = 1
        return rows, prototypes

    def test_instance_mask_and_box_crop_are_decoded(self):
        model = self.model()
        detections = model.decode(self.outputs(), (480, 640, 3))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].region, (240, 90, 160, 300))
        self.assertTrue(detections[0].mask[200, 300])
        self.assertFalse(detections[0].mask[200, 239])
        self.assertFalse(detections[0].mask[390, 300])

    def test_portrait_padding_is_removed_from_mask_and_coordinates(self):
        model = self.model()
        detections = model.decode(self.outputs(), (480, 320, 3), padding=(160, 0), resized=(320, 480))
        self.assertEqual(detections[0].region, (80, 90, 160, 300))
        self.assertEqual(detections[0].mask.shape, (480, 320))
        self.assertTrue(detections[0].mask[200, 150])

    def test_person_is_not_accepted_as_bottle(self):
        rows, prototypes = self.outputs()
        rows[0, 4, 0] = .99
        self.assertEqual(self.model().decode((rows, prototypes), (480, 640, 3)), [])

    def test_rgb_normalization_and_letterbox(self):
        model = self.model()
        model.net.forward.return_value = self.outputs()
        model.detect(np.full((480, 320, 3), (10, 20, 30), np.uint8))
        blob = model.net.setInput.call_args.args[0]
        np.testing.assert_allclose(blob[0, :, 50, 200], np.array([30, 20, 10]) / 255, atol=1e-7)
        np.testing.assert_allclose(blob[0, :, 50, 10], np.full(3, 114 / 255), atol=1e-7)


class LearnedFinderTests(unittest.TestCase):
    def test_full_bottle_candidate_is_verified_without_switching_to_hand_crop(self):
        sizes = []

        def detect(frame):
            sizes.append(frame.shape[:2])
            return [Detection((255, 48, 130, 371), .9)]

        finder = LearnedBottleFinder(SimpleNamespace(detect=detect))
        frame, _ = axis_frame()
        hand = [(320, 250)] * 21
        self.assertIsNone(finder.update(frame, hand, 0))
        self.assertIsNotNone(finder.update(frame, hand, 2))
        self.assertEqual(sizes, [(480, 640), (480, 640)])

    def test_colored_background_strip_cannot_replace_cap(self):
        frame = np.full((480, 640, 3), 160, np.uint8)
        cv2.rectangle(frame, (236, 85), (265, 160), (20, 110, 110), -1)
        cv2.rectangle(frame, (281, 86), (321, 116), (190, 60, 130), -1)
        cap, _, _ = propose_endpoints(frame, Detection((236, 85, 112, 309), .5), [])
        np.testing.assert_allclose(cap, (301, 101), atol=2)

    def test_neck_only_detection_never_becomes_bottle_base(self):
        finder = LearnedBottleFinder(SimpleNamespace(detect=lambda _: [Detection((254, 84, 91, 150), .8)]), stable_samples=1)
        candidate = finder.update(np.full((480, 640, 3), 150, np.uint8), [], 0)
        self.assertIsNone(candidate)
        self.assertIsNotNone(finder._focus_region)
        self.assertIn('whole bottle', finder.status)

    def test_neck_fragment_and_small_background_bottles_do_not_block_full_bottle(self):
        # Reproduces the actual scene: neck score > full bottle score, plus
        # small bottles on a background shelf. Ordinary IoU NMS keeps them all.
        full = Detection((218, 84, 128, 318), .259)
        detections = [Detection((254, 84, 91, 150), .414),
                      Detection((128, 245, 36, 66), .357), full]
        finder = LearnedBottleFinder(SimpleNamespace(detect=lambda _: detections), stable_samples=1)
        candidate = finder.update(np.full((480, 640, 3), 150, np.uint8), [], 0)
        self.assertIs(finder.detection, full)
        self.assertIsNotNone(candidate)
        self.assertGreater(np.linalg.norm(candidate.base - candidate.cap), 250)

    def test_throttled_search_does_not_publish_empty_result_over_completed_candidate(self):
        detector = SimpleNamespace(detect=lambda _: [Detection((255, 48, 130, 371), .9)])
        processor = BottleSearchProcessor(detector, 'test')
        frame, _ = axis_frame()

        def packet(index, now):
            return SimpleNamespace(frame_id=index, captured_at=now, frame=frame, hand=[], expected=None, generation=0)

        first = processor.process(packet(0, 0), [])
        self.assertIsNotNone(first)
        self.assertIsNone(processor.process(packet(1, .2), []))
        complete = processor.process(packet(2, 1.2), [])
        self.assertIsNotNone(complete['candidate'])
        self.assertIsNone(processor.process(packet(3, 1.3), []))

    def test_localization_does_not_require_visible_rim_or_measurable_width(self):
        frame = np.full((480, 640, 3), 165, np.uint8)
        detector = SimpleNamespace(detect=lambda _: [Detection((220, 60, 170, 340), .8)])
        finder = LearnedBottleFinder(detector, interval=.1, stable_samples=2)
        with patch('bottle_axis.detect_landmarks', side_effect=AssertionError('No global geometric search')):
            self.assertIsNone(finder.acquire(frame, [], 0))
            tracker = finder.acquire(frame, [], .2)
            self.assertIsNotNone(tracker)
            self.assertIsNone(tracker.last.width_px)
            self.assertFalse(tracker.endpoint_tracker.allow_geometry)
            self.assertEqual(tracker.source, 'learned_bottle_box')

    def test_mask_seed_does_not_treat_bounding_box_as_deformation(self):
        frame, _ = axis_frame()
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.rectangle(mask, (285, 48), (355, 82), 1, -1)
        cv2.rectangle(mask, (255, 112), (385, 418), 1, -1)
        seed = detection_landmarks(frame, InstanceDetection((255, 48, 130, 371), .9, mask), [])
        self.assertEqual(seed.source, 'learned_instance_mask')
        self.assertAlmostEqual(seed.cap[0], 320, delta=4)
        self.assertAlmostEqual(seed.base[0], 320, delta=4)

    def test_multiple_bottles_without_hand_are_not_arbitrarily_selected(self):
        detector = SimpleNamespace(detect=lambda _: [Detection((40, 60, 100, 320), .8), Detection((400, 60, 100, 320), .8)])
        finder = LearnedBottleFinder(detector)
        self.assertIsNone(finder.acquire(np.zeros((480, 640, 3), np.uint8), [], 0))
        self.assertIsNone(finder.detection)


class WorkerTests(unittest.TestCase):
    def test_debug_snapshot_keeps_camera_and_analyzed_source_frames_separate(self):
        camera = np.full((480, 640, 3), 60, np.uint8)
        analyzed = np.full_like(camera, 20)
        packet = FramePacket(12, 5., camera)
        result = {'canvas': np.zeros((650, 1000, 3), np.uint8), 'analysis_frame': analyzed,
                  'frame_id': 8, 'captured_at': 4.8, 'processing_ms': 45.,
                  'axis_detected': True, 'baseline_ready': False, 'bottle_state': 'tracked',
                  'left_profiles': 1, 'right_profiles': 9}
        with tempfile.TemporaryDirectory() as directory:
            path = save_debug_snapshot(Path(directory), packet, result, 5.)
            state = json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(state['camera_frame_id'], 12)
            self.assertEqual(state['analysis']['frame_id'], 8)
            self.assertAlmostEqual(state['analysis_age_ms'], 200.)
            self.assertTrue(state['analysis']['axis_detected'])
            np.testing.assert_array_equal(cv2.imread(str(path)), camera)
            np.testing.assert_array_equal(cv2.imread(str(path.with_name(path.stem + '_analysis.png'))), analyzed)
            self.assertTrue(path.with_name(path.stem + '_preview.png').exists())

    def test_no_work_tick_keeps_previous_completed_result(self):
        first, skipped = Event(), Event()

        class Processor:
            def process(self, packet, commands):
                if packet.frame_id == 0:
                    first.set()
                    return {'frame_id': 0}
                skipped.set()
                return None

            def close(self):
                pass

        worker = LatestWorker(Processor, 30).start()
        try:
            worker.submit(FramePacket(0, 0, None))
            self.assertTrue(first.wait(1))
            worker.submit(FramePacket(1, .1, None))
            self.assertTrue(skipped.wait(1))
        finally:
            worker.close()
        self.assertEqual(worker.read_latest(), {'frame_id': 0})

    def test_slow_analysis_discards_backlog_and_keeps_commands(self):
        entered, release, finished = Event(), Event(), Event()
        seen = []

        class Processor:
            closed = False

            def process(self, packet, commands):
                seen.append((packet.frame_id, commands))
                if packet.frame_id == 0:
                    entered.set()
                    release.wait(2)
                if packet.frame_id == 8:
                    finished.set()
                return {'frame_id': packet.frame_id}

            def close(self):
                self.closed = True

        processor = Processor()
        worker = LatestWorker(lambda: processor, max_fps=30).start()
        try:
            worker.submit(FramePacket(0, time.monotonic(), np.zeros((2, 2, 3), np.uint8)))
            self.assertTrue(entered.wait(1))
            for frame_id in range(1, 9):
                worker.submit(FramePacket(frame_id, time.monotonic(), np.zeros((2, 2, 3), np.uint8)))
            worker.command('r')
            worker.command('r')
            release.set()
            self.assertTrue(finished.wait(2))
        finally:
            release.set()
            worker.close()
        self.assertEqual([v[0] for v in seen], [0, 8])
        self.assertEqual(seen[-1][1], ['r', 'r'])
        self.assertEqual(worker.dropped, 7)
        self.assertTrue(processor.closed)

    def test_fresh_camera_replaces_old_pixels_and_stale_overlay_is_hidden(self):
        frame = np.full((480, 640, 3), (10, 20, 30), np.uint8)
        canvas = np.zeros((650, 1000, 3), np.uint8)
        canvas[100, 100] = (255, 255, 255)
        canvas[100, 800] = (40, 50, 60)
        result = {'canvas': canvas, 'captured_at': 4., 'processing_ms': 500}
        packet = FramePacket(9, 5., frame)
        composed = compose_preview(packet, result, 5.)
        np.testing.assert_array_equal(composed[100, 100], frame[100, 100])
        np.testing.assert_array_equal(composed[100, 800], canvas[100, 800])
        fresh = compose_preview(packet, {**result, 'captured_at': 4.9}, 5.)
        np.testing.assert_array_equal(fresh[100, 100], [255, 255, 255])

    def test_camera_is_visible_while_models_load(self):
        frame = np.full((480, 640, 3), 128, np.uint8)
        result = compose_preview(FramePacket(0, 0., frame), None, 0., loading=True)
        np.testing.assert_array_equal(result[100, 100], [128, 128, 128])


class RealtimeSessionTests(unittest.TestCase):
    def test_lost_uncalibrated_axis_restarts_search_but_calibrated_reference_is_kept(self):
        for calibrated in (False, True):
            with self.subTest(calibrated=calibrated):
                hand = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(hand_landmarks=[]), close=Mock())
                session = RealtimeSession(hand, None, Path('unused'), {}, 'test')
                frame, _ = axis_frame()
                band = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
                session.band = band
                if calibrated:
                    session.baseline.distance = 130.
                session.finder.update = Mock(return_value=None)
                base = time.monotonic()
                try:
                    with patch.object(band.endpoint_tracker, 'track', return_value=(None, 0., 'Axis lost')):
                        session.process(FramePacket(0, base, frame), [])
                        session.process(FramePacket(1, base + 1., frame), [])
                    if calibrated:
                        self.assertIs(session.band, band)
                        self.assertEqual(session.baseline.distance, 130.)
                    else:
                        self.assertIsNone(session.band)
                        self.assertIsNone(session.calibration)
                        self.assertIn('Reacquiring', session.status)
                finally:
                    session.close()

    def test_slow_bottle_inference_does_not_block_hand_analysis(self):
        entered, release = Event(), Event()

        def detect(frame):
            entered.set()
            release.wait(3)
            return [Detection((255, 48, 130, 371), .9)]

        hand = SimpleNamespace(detect_for_video=Mock(return_value=SimpleNamespace(hand_landmarks=[])), close=Mock())
        session = RealtimeSession(hand, None, Path('unused'), {}, 'loading',
                                  search_factory=lambda: BottleSearchProcessor(SimpleNamespace(detect=detect), 'test'))
        base = time.monotonic()
        frame, _ = axis_frame()
        try:
            session.process(FramePacket(0, base, frame), [])
            self.assertTrue(entered.wait(1))
            # The bottle model is still blocked throughout these hand updates.
            for index in range(1, 7):
                session.process(FramePacket(index, base + index / 12, frame), [])
            self.assertEqual(hand.detect_for_video.call_count, 7)
            self.assertFalse(release.is_set())
            self.assertIsNone(session.finder.worker.error)
        finally:
            release.set()
            session.close()
        hand.close.assert_called_once()

    def test_hand_crop_maps_points_to_original_frame_and_keeps_own_tracking_state(self):
        missing = SimpleNamespace(hand_landmarks=[])
        point = SimpleNamespace(x=.5, y=.5, z=0.)
        found = SimpleNamespace(hand_landmarks=[[point] * 21],
                                handedness=[[SimpleNamespace(category_name='Right')]],
                                hand_world_landmarks=[[point] * 21])
        full = Mock(return_value=missing)
        crop = Mock(return_value=found)
        landmarker = SimpleNamespace(detect_for_video=full, roi_model=SimpleNamespace(detect_for_video=crop),
                                     prefer_roi=False, roi_view=None)
        frame = np.zeros((480, 640, 3), np.uint8)
        hand = observe_hand(frame, landmarker, 100, (200, 50, 100, 300))
        self.assertEqual(hand.points, [(250, 200)] * 21)
        self.assertTrue(landmarker.prefer_roi)
        self.assertEqual(full.call_args.args[1], 300)
        self.assertEqual(crop.call_args.args[1], 301)
        self.assertEqual(crop.call_args.args[0].numpy_view().shape[:2], (342, 342))
        # The asynchronous detector and tracked axis need not return identical
        # boxes. Small changes must not resize/recenter the hand video stream.
        hand = observe_hand(frame, landmarker, 101, (206, 52, 100, 294))
        self.assertEqual(hand.points, [(250, 200)] * 21)
        self.assertEqual(crop.call_args.args[0].numpy_view().shape[:2], (342, 342))
        self.assertEqual(full.call_count, 1)
        self.assertEqual(crop.call_count, 2)
        hand = observe_hand(frame, landmarker, 102, (360, 60, 110, 300))
        self.assertEqual(hand.points, [(415, 210)] * 21)
        self.assertEqual(crop.call_count, 3)

    def test_csv_uses_unique_analyzed_frame_ids_and_capture_times(self):
        landmarker = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(hand_landmarks=[]), close=Mock())
        detector = SimpleNamespace(detect=lambda _: [Detection((255, 48, 130, 371), .9)])
        with tempfile.TemporaryDirectory() as directory:
            session = RealtimeSession(landmarker, detector, Path(directory), {'reference_grip_kgf': 31.5}, 'fake_test_model')
            base = time.monotonic() - 10
            for index in range(82):
                frame, _ = axis_frame(dent=0 if index < 74 else .2)
                commands = ['r'] if index in (70, 81) else []
                session.process(FramePacket(index, base + index / 20, frame), commands)
            session.close()
            path = next(Path(directory).glob('*.csv'))
            with path.open(newline='', encoding='utf-8') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([int(r['source_frame_id']) for r in rows], list(range(70, 81)))
            self.assertTrue(all(None not in r for r in rows))
            self.assertEqual(rows[0]['reference_grip_kgf'], '31.5')
            self.assertGreater(float(rows[-1]['dent_p95_pct']), 12)
            self.assertAlmostEqual(float(rows[-1]['capture_time_s']) - float(rows[0]['capture_time_s']), .5)
            landmarker.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

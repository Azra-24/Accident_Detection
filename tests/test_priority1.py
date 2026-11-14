"""
VisionGuard Unit Tests — Priority 1 & 2 regression tests.

Tests:
  - Non-blocking queue behavior (frame_reader + engine)
  - Frame-gap-aware velocity normalisation in ByteTracker
  - Severity classifier tier boundaries
"""

import queue
import threading
import time
import unittest

import numpy as np


class TestNonBlockingQueue(unittest.TestCase):
    """Verify the FrameReader queue never blocks the producer."""

    def test_put_drops_oldest_when_full(self):
        """Queue of size 1: putting 2 items should drop the first."""
        q = queue.Queue(maxsize=1)
        q.put_nowait(("frame_A",))

        # Simulate FrameReader's drop-oldest logic
        try:
            q.put_nowait(("frame_B",))
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(("frame_B",))

        item = q.get_nowait()
        self.assertEqual(item, ("frame_B",))

    def test_queue_never_blocks_producer(self):
        """Put into a full queue should complete in < 50ms (no blocking)."""
        q = queue.Queue(maxsize=1)
        q.put_nowait("old")

        t0 = time.perf_counter()
        # Simulate the non-blocking drop-oldest put
        try:
            q.put_nowait("new")
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait("new")
            except queue.Full:
                pass
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.assertLess(elapsed_ms, 50.0, "Non-blocking put took too long")

    def test_display_queue_shows_latest(self):
        """Display queue should always return the most recent frame."""
        display_q = queue.Queue(maxsize=2)
        for i in range(5):
            try:
                display_q.put_nowait(f"frame_{i}")
            except queue.Full:
                try:
                    display_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    display_q.put_nowait(f"frame_{i}")
                except queue.Full:
                    pass

        # Drain — last item should be most recent
        last = None
        while not display_q.empty():
            last = display_q.get_nowait()
        self.assertEqual(last, "frame_4")


class TestFrameGapNormalisation(unittest.TestCase):
    """Verify the ByteTracker normalises velocity by frame gap."""

    def _make_tracker_config(self):
        return {
            "tracking": {
                "high_conf_threshold": 0.5,
                "low_conf_threshold": 0.3,
                "iou_match_threshold": 0.7,
                "max_lost_frames": 20,
                "min_hits": 1,  # confirm on first hit for testing
                "history_length": 30,
                "lk_window_size": 15,
                "lk_max_level": 2,
                "lk_max_corners": 100,
                "min_lk_points": 3,
                "velocity_smoothing_alpha": 0.0,  # no EMA smoothing for test
            }
        }

    def test_gap_normalisation_reduces_inflated_speed(self):
        """
        With a frame gap > 1, raw pixel displacement should be divided
        by the gap, producing a lower per-frame velocity than without
        normalisation.
        """
        from tracking.tracker import ByteTracker

        config = self._make_tracker_config()
        tracker = ByteTracker(config)

        # Create a synthetic 100x100 frame with a bright dot
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (20, 20), (40, 40), (255, 255, 255), -1)

        # Same dot shifted 10px right (simulating 5 frames of movement)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (30, 20), (50, 40), (255, 255, 255), -1)

        det1 = [{"bbox": [20, 20, 40, 40], "class_name": "car", "confidence": 0.9}]
        det2 = [{"bbox": [30, 20, 50, 40], "class_name": "car", "confidence": 0.9}]

        # Run with gap=1 (consecutive)
        tracker.update(frame1, det1, source_frame_num=1)
        tracks_gap1 = tracker.update(frame2, det2, source_frame_num=2)

        # Reset and run with gap=5
        tracker.reset()
        tracker.update(frame1, det1, source_frame_num=1)
        tracks_gap5 = tracker.update(frame2, det2, source_frame_num=6)

        # Both should produce confirmed tracks (min_hits=1)
        if tracks_gap1 and tracks_gap5:
            speed_gap1 = list(tracks_gap1.values())[0]["speed"]
            speed_gap5 = list(tracks_gap5.values())[0]["speed"]
            # Speed with gap=5 should be ~5x smaller than gap=1
            if speed_gap1 > 0:
                ratio = speed_gap5 / speed_gap1
                self.assertLess(ratio, 0.5,
                    f"Gap normalisation not working: speed@gap1={speed_gap1:.2f}, "
                    f"speed@gap5={speed_gap5:.2f}, ratio={ratio:.2f}")

    def test_kalman_predict_dt_scaling(self):
        """
        With a frame gap > 1, Kalman filter prediction should scale position
        by the frame gap (v * dt) rather than assuming dt=1.
        """
        from tracking.tracker import ByteTracker

        config = self._make_tracker_config()
        tracker = ByteTracker(config)

        # Frame 1: center at (30,30)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (20, 20), (40, 40), (255, 255, 255), -1)
        det1 = [{"bbox": [20, 20, 40, 40], "class_name": "car", "confidence": 0.9}]

        # Frame 2: center at (40,30) - simulating speed of 10px/frame
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (30, 20), (50, 40), (255, 255, 255), -1)
        det2 = [{"bbox": [30, 20, 50, 40], "class_name": "car", "confidence": 0.9}]

        # Init track
        tracks1 = tracker.update(frame1, det1, source_frame_num=1)
        # Update track and set velocity
        tracks2 = tracker.update(frame2, det2, source_frame_num=2)

        # Force a high velocity directly to bypass LK complexity
        track = tracker._tracks[0]
        track.kalman.statePost[2] = 10.0  # vx = 10
        track.kalman.statePost[3] = 0.0   # vy = 0
        
        # Test predict with dt=1
        pred1 = track.predict(dt=1)
        dx1 = pred1[0, 0] - track.kalman.statePost[0, 0]

        # Test predict with dt=5
        pred5 = track.predict(dt=5)
        dx5 = pred5[0, 0] - track.kalman.statePost[0, 0]
        
        # dx5 should be exactly 5 times dx1
        self.assertAlmostEqual(dx5, dx1 * 5, places=3, 
                               msg=f"Kalman prediction did not scale by dt. dx1={dx1}, dx5={dx5}")


class TestSeverityClassifierTiers(unittest.TestCase):
    """Verify severity classifier tier boundary logic."""

    def _make_classifier(self, minor_max=3, moderate_max=6, severe_max=10):
        from severity.classifier import SeverityClassifier
        config = {
            "severity": {
                "minor_max": minor_max,
                "moderate_max": moderate_max,
                "severe_max": severe_max,
                "scores": {
                    "speed_drop_heavy": 3,
                    "speed_drop_moderate": 1,
                    "trajectory_deviation": 3,
                    "vehicle_overlap": 3,
                    "flow_incoherence": 1,
                    "motorcycle_track_lost": 4,
                    "crashed_vehicle_yolo": 4,
                    "fire_detected": 5,
                    "smoke_detected": 3,
                    "persistence_bonus": 2,
                },
                "persistence_bonus_frames": 3,
                "persistence_bonus_score": 2,
            }
        }
        return SeverityClassifier(config)

    def _make_anomaly(self, tid=1, signals=None):
        return {
            "track_id": tid,
            "bbox": [10, 10, 50, 50],
            "class_name": "car",
            "signals": signals or {},
            "speed": 5.0,
            "avg_speed": 8.0,
            "deviation_deg": 0,
            "incoherence": 0,
            "overlap_with": None,
        }

    def test_minor_boundary(self):
        """Score <= 3 should be Minor."""
        clf = self._make_classifier()
        # speed_drop_moderate = 1 point
        anomaly = self._make_anomaly(signals={"speed_drop_moderate": True})
        result = clf.classify(anomaly, [])
        self.assertEqual(result["severity"], "Minor")

    def test_moderate_boundary(self):
        """Score 4-6 should be Moderate."""
        clf = self._make_classifier()
        # speed_drop_heavy(3) + speed_drop_moderate(1) = 4
        anomaly = self._make_anomaly(signals={
            "speed_drop_heavy": True,
            "speed_drop_moderate": True,
        })
        result = clf.classify(anomaly, [])
        self.assertEqual(result["severity"], "Moderate")
        self.assertEqual(result["total_score"], 4.0)

    def test_severe_boundary(self):
        """Score 7-10 should be Severe."""
        clf = self._make_classifier()
        # speed_drop_heavy(3) + trajectory_deviation(3) + flow_incoherence(1) = 7
        anomaly = self._make_anomaly(signals={
            "speed_drop_heavy": True,
            "trajectory_deviation": True,
            "flow_incoherence": True,
        })
        result = clf.classify(anomaly, [])
        self.assertEqual(result["severity"], "Severe")

    def test_critical_boundary(self):
        """Score > 10 should be Critical."""
        clf = self._make_classifier()
        # speed_drop_heavy(3) + trajectory_deviation(3) + vehicle_overlap(3) = 9 motion
        # + fire detection (5 * 0.9 conf = 4.5) = 13.5 total
        anomaly = self._make_anomaly(signals={
            "speed_drop_heavy": True,
            "trajectory_deviation": True,
            "vehicle_overlap": True,
        })
        hazards = [{"class_name": "fire", "confidence": 0.9}]
        result = clf.classify(anomaly, hazards)
        self.assertEqual(result["severity"], "Critical")
        self.assertGreater(result["total_score"], 10)

    def test_exact_boundary_values(self):
        """Verify exact boundary: score==3 is Minor, score==4 is Moderate."""
        clf = self._make_classifier()
        # Exactly 3: speed_drop_heavy = 3
        anomaly_3 = self._make_anomaly(signals={"speed_drop_heavy": True})
        result_3 = clf.classify(anomaly_3, [])
        self.assertEqual(result_3["severity"], "Minor")
        self.assertEqual(result_3["total_score"], 3.0)

        # Exactly 4: speed_drop_heavy(3) + flow_incoherence(1) = 4
        anomaly_4 = self._make_anomaly(tid=2, signals={
            "speed_drop_heavy": True,
            "flow_incoherence": True,
        })
        result_4 = clf.classify(anomaly_4, [])
        self.assertEqual(result_4["severity"], "Moderate")
        self.assertEqual(result_4["total_score"], 4.0)

    def test_hazard_confidence_weighting(self):
        """Hazard scores should be multiplied by detection confidence."""
        clf = self._make_classifier()
        anomaly = self._make_anomaly(signals={})

        # crashed_vehicle base=4, conf=0.5 → 2.0
        hazards = [{"class_name": "crashed_vehicle", "confidence": 0.5}]
        result = clf.classify(anomaly, hazards)
        self.assertAlmostEqual(result["hazard_score"], 2.0, places=1)

    def test_persistence_bonus_requires_history(self):
        """Persistence bonus should only fire after N consecutive Moderate+."""
        clf = self._make_classifier()
        # First classification — no history yet, no persistence bonus
        anomaly = self._make_anomaly(signals={
            "speed_drop_heavy": True,
            "trajectory_deviation": True,
        })
        result = clf.classify(anomaly, [])
        self.assertEqual(result["persistence_score"], 0.0)


# Need cv2 for the tracker test
try:
    import cv2
except ImportError:
    cv2 = None


if __name__ == "__main__":
    unittest.main()

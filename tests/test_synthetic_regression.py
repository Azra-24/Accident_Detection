"""
Synthetic Regression Test Suite for VisionGuard
Tests tracking and anomaly logic using mathematically defined trajectories.
"""

import unittest
import numpy as np
import cv2
import yaml

from tracking.tracker import ByteTracker
from anomaly.detector import AnomalyDetector
from severity.classifier import SeverityClassifier


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


class SyntheticTrajectoryRunner:
    def __init__(self, config):
        self.tracker = ByteTracker(config)
        self.anomaly_det = AnomalyDetector(config)
        self.classifier = SeverityClassifier(config)
        # Ensure fast confirmation for tests
        self.tracker._min_hits = 1
        # Set max_lost lower for occlusion test
        self.tracker._max_lost = 5
        self.events = []

    def run_trajectory(self, trajectory):
        """
        trajectory: list of dicts with 'bbox', 'drop_frame'
        """
        source_frame = 0
        for step in trajectory:
            source_frame += step.get("gap", 1)
            
            # Generate frame
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            bbox = step.get("bbox")
            
            det = []
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
                det = [{"bbox": bbox, "class_name": step.get("class_name", "car"), "confidence": 0.9}]

            # Process
            tracks = self.tracker.update(frame, det, source_frame_num=source_frame)
            anomalies = self.anomaly_det.detect(tracks, source_frame)
            
            for a in anomalies:
                event = self.classifier.classify(a, [])
                self.events.append((source_frame, event))
                
        return self.events


class TestSyntheticTrajectories(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.runner = SyntheticTrajectoryRunner(self.config)

    def _create_straight_line(self, start_x, speed, length, gap=1):
        traj = []
        for i in range(length):
            x = start_x + (i * speed * gap)
            traj.append({"bbox": [x, 50, x+20, 70], "gap": gap})
        return traj

    # --- NORMAL CASES ---
    def test_normal_constant_speed(self):
        """Cruising at constant speed should produce no events."""
        traj = self._create_straight_line(10, 3.0, 30)
        events = self.runner.run_trajectory(traj)
        self.assertEqual(len(events), 0)

    def test_normal_slow_speed(self):
        """Cruising slowly should not trigger speed drop."""
        traj = self._create_straight_line(10, 1.0, 30)
        events = self.runner.run_trajectory(traj)
        self.assertEqual(len(events), 0)

    def test_normal_fast_speed(self):
        """Cruising fast should produce no events."""
        traj = self._create_straight_line(10, 8.0, 30)
        events = self.runner.run_trajectory(traj)
        self.assertEqual(len(events), 0)

    def test_normal_minor_jitter(self):
        """Small position jitter should not trigger deviation or incoherence."""
        traj = []
        x = 10
        for i in range(30):
            jitter = (i % 2) * 2 - 1  # -1, 1, -1, 1
            x += 4.0
            traj.append({"bbox": [x, 50 + jitter, x+20, 70 + jitter], "gap": 1})
        events = self.runner.run_trajectory(traj)
        self.assertEqual(len(events), 0)

    def test_normal_acceleration(self):
        """Gradual acceleration should be fine."""
        traj = []
        x = 10
        speed = 2.0
        for i in range(30):
            x += speed
            speed += 0.2
            traj.append({"bbox": [x, 50, x+20, 70], "gap": 1})
        events = self.runner.run_trajectory(traj)
        self.assertEqual(len(events), 0)

    # --- TRIGGER CASES ---
    def test_trigger_sudden_deceleration(self):
        """Dropping speed suddenly should trigger speed_drop_heavy."""
        traj = self._create_straight_line(10, 5.0, 20)
        # suddenly drop speed to 1.0
        x = traj[-1]["bbox"][0]
        for i in range(10):
            x += 1.0
            traj.append({"bbox": [x, 50, x+20, 70], "gap": 1})
            
        events = self.runner.run_trajectory(traj)
        self.assertTrue(len(events) > 0)
        signals = events[0][1]["all_signals"]
        self.assertTrue("speed_drop_heavy" in signals or "speed_drop_moderate" in signals)

    def test_trigger_erratic_swerving(self):
        """90-degree turn should trigger trajectory_deviation."""
        traj = self._create_straight_line(10, 4.0, 15)
        # turn 90 degrees down
        x = traj[-1]["bbox"][0]
        y = 50
        for i in range(15):
            y += 4.0
            traj.append({"bbox": [x, y, x+20, y+20], "gap": 1})
            
        events = self.runner.run_trajectory(traj)
        self.assertTrue(len(events) > 0)
        
        # Could be trajectory deviation or incoherence
        has_signal = False
        for _, event in events:
            if "trajectory_deviation" in event["all_signals"]:
                has_signal = True
                break
        self.assertTrue(has_signal)

    def test_trigger_vehicle_overlap(self):
        """Two boxes merging into one space should trigger overlap."""
        traj = []
        # Simulate two tracks closing in on each other
        for i in range(20):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            x1, y1 = 10 + i * 3, 50
            x2, y2 = 100 - i * 3, 50
            cv2.rectangle(frame, (x1, y1), (x1+20, y1+20), (255, 255, 255), -1)
            cv2.rectangle(frame, (x2, y2), (x2+20, y2+20), (255, 255, 255), -1)
            
            det = [
                {"bbox": [x1, y1, x1+20, y1+20], "class_name": "car", "confidence": 0.9},
                {"bbox": [x2, y2, x2+20, y2+20], "class_name": "car", "confidence": 0.9}
            ]
            tracks = self.runner.tracker.update(frame, det, source_frame_num=i)
            anomalies = self.runner.anomaly_det.detect(tracks, i)
            for a in anomalies:
                event = self.runner.classifier.classify(a, [])
                self.runner.events.append((i, event))
                
        # Overlap should be detected
        overlap_events = [e for e in self.runner.events if "vehicle_overlap" in e[1]["all_signals"]]
        self.assertTrue(len(overlap_events) > 0)

    # --- EDGE CASES ---
    def test_edge_brief_occlusion(self):
        """Track should survive brief disappearance."""
        # Using a realistic car bounding box width (50px) so IoU survives a brief 3-frame occlusion
        traj = []
        x = 10
        for _ in range(15):
            x += 4.0
            traj.append({"bbox": [x, 50, x+50, 80], "gap": 1})
        
        # Missing detection for 3 frames (bbox is None)
        for _ in range(3):
            x += 4.0
            traj.append({"bbox": None, "gap": 1})
            
        # Reappears
        for _ in range(10):
            x += 4.0
            traj.append({"bbox": [x, 50, x+50, 80], "gap": 1})
            
        self.runner.run_trajectory(traj)
        # Should be a single continuous track with id 0
        self.assertIn(0, self.runner.tracker._tracks)

    def test_edge_simulated_frame_drops(self):
        """Frame gaps should not falsely trigger deceleration."""
        traj = []
        x = 10
        speed = 4.0
        # First 10 frames normal
        for i in range(10):
            x += speed
            traj.append({"bbox": [x, 50, x+20, 70], "gap": 1})
            
        # Simulate a 5-frame gap
        x += speed * 5
        traj.append({"bbox": [x, 50, x+20, 70], "gap": 5})
        
        # Continue normal
        for i in range(10):
            x += speed
            traj.append({"bbox": [x, 50, x+20, 70], "gap": 1})
            
        events = self.runner.run_trajectory(traj)
        # Normalization should prevent speed_drop anomaly
        self.assertEqual(len(events), 0)


if __name__ == "__main__":
    unittest.main()

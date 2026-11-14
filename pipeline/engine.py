"""
VisionGuard Pipeline Engine — Dual-threaded, non-blocking architecture.

Two threads:
  Thread 1 (Detection): Reads from frame_queue, runs full pipeline
           (motion gate → YOLO-A → ByteTrack+LK → anomaly → YOLO-B
           → severity → annotate), puts result into display_queue.

  Main Thread (Display): Reads from source, feeds frame_queue,
           reads display_queue, shows frames via cv2.imshow.
           NEVER blocks on detection.

Frame drops are silent and acceptable. Display lag is never visible.
Graceful shutdown via threading.Event stop_event.
"""

import logging
import queue
import threading
import time

import cv2
import numpy as np
import yaml

from preprocessing import AdaptivePreprocessor
from detection import VehicleDetector, HazardDetector
from tracking import ByteTracker
from anomaly import AnomalyDetector
from severity import SeverityClassifier
from reporting import EventLogger
from alerts import TelegramAlert

logger = logging.getLogger(__name__)


class VisionGuardEngine:
    """
    Master pipeline engine with non-blocking threaded architecture.

    The display thread NEVER waits for detection. If detection is slow,
    frames are dropped silently and the last known annotated frame is
    displayed.
    """

    SEVERITY_COLORS = {
        "Minor": (0, 255, 255),       # yellow
        "Moderate": (0, 165, 255),     # orange
        "Severe": (0, 0, 255),         # red
        "Critical": (0, 0, 180),       # dark red
    }

    HAZARD_COLORS = {
        "crashed_vehicle": (0, 0, 255),
        "fire": (0, 100, 255),
        "smoke": (200, 200, 200),
    }

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        pcfg = self.config.get("pipeline", {})

        # Pipeline modules
        self.preprocessor = AdaptivePreprocessor(self.config)
        self.vehicle_det = VehicleDetector(self.config)
        self.hazard_det = HazardDetector(self.config)
        self.tracker = ByteTracker(self.config)
        self.anomaly_det = AnomalyDetector(self.config)
        self.classifier = SeverityClassifier(self.config)
        self.logger = EventLogger(self.config)
        self.alerter = TelegramAlert(self.config)

        # Pipeline settings
        self._detect_interval: int = pcfg.get("detect_interval", 3)
        self._log_minor: bool = pcfg.get("log_minor", False)
        self._display_w: int = pcfg.get("display_width", 1280)
        self._display_h: int = pcfg.get("display_height", 720)
        self._queue_size: int = pcfg.get("queue_size", 4)

        # Alert settings
        acfg = self.config.get("alerts", {})
        self._min_signals: int = acfg.get("min_signals_to_alert", 2)

        # Threading
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self._queue_size)
        self._display_queue: queue.Queue = queue.Queue(maxsize=self._queue_size)

        # State
        self.frame_count: int = 0
        self._cached_display_frame: np.ndarray = None
        self._cached_vehicle_dets: list = []
        self._active_events: dict = {}
        self._fps_counter: float = 0.0
        self._fps_timer: float = 0.0
        self._last_source_frame: int = 0
        self._total_frame_gap: int = 0
        self._max_frame_gap: int = 0

        logger.info(
            "VisionGuardEngine initialised: interval=%d, queue_size=%d",
            self._detect_interval,
            self._queue_size,
        )

    # ══════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════

    def run_video(self, video_path: str, show_window: bool = True,
                  simulate_live_fps: bool = False) -> None:
        """
        Process a video file with non-blocking display.

        Parameters
        ----------
        video_path : str
            Path to the input video file.
        show_window : bool
            Whether to show the cv2 display window.
        simulate_live_fps : bool
            If True, pace frame reads at the source video's native FPS
            to simulate live camera behavior for benchmarking.
        """
        from pipeline.frame_reader import FrameReader
        
        # Start background frame reader
        reader = FrameReader(
            video_path,
            queue_size=self._queue_size,
            stop_event=self._stop_event,
            simulate_live_fps=simulate_live_fps,
        )
        reader.start()

        fps = reader.fps
        total = reader.total_frames
        orig_w = reader.width
        orig_h = reader.height

        # Frame delay for normal-speed playback (ms)
        frame_delay_ms = max(1, int(1000.0 / fps))

        print(f"[VisionGuard] Processing: {video_path}")
        print(
            f"[VisionGuard] FPS: {fps:.1f} | Frames: {total} | "
            f"Resolution: {orig_w}x{orig_h}"
        )

        if show_window:
            cv2.namedWindow("VisionGuard", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("VisionGuard", self._display_w, self._display_h)
            cv2.setWindowProperty("VisionGuard", cv2.WND_PROP_TOPMOST, 1)

        # Reset state
        self._stop_event.clear()
        self.frame_count = 0
        self._fps_timer = time.perf_counter()

        # Connect detection thread to the reader's queue
        self._frame_queue = reader._queue

        # Start detection thread
        det_thread = threading.Thread(
            target=self._detection_loop,
            name="DetectionThread",
            daemon=True,
        )
        det_thread.start()
        logger.info("Detection thread started")

        dropped_display_frames = 0
        total_display_frames = 0

        # ── Main thread: display only ────────────────────
        try:
            while not self._stop_event.is_set():
                frame_start = time.perf_counter()

                # Non-blocking get from display queue
                try:
                    display_frame = self._display_queue.get_nowait()
                    self._cached_display_frame = display_frame
                    total_display_frames += 1
                except queue.Empty:
                    display_frame = self._cached_display_frame
                    dropped_display_frames += 1

                if display_frame is not None and show_window:
                    resized = cv2.resize(
                        display_frame, (self._display_w, self._display_h)
                    )
                    cv2.imshow("VisionGuard", resized)

                # FPS throttle — wait remaining time to match source FPS
                elapsed_ms = (time.perf_counter() - frame_start) * 1000
                wait_ms = max(1, frame_delay_ms - int(elapsed_ms))

                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    logger.info("User pressed 'q' — stopping")
                    self._stop_event.set()
                    break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — stopping")
            self._stop_event.set()
        finally:
            reader.stop()
            # Wait for detection thread to finish
            self._stop_event.set()
            det_thread.join(timeout=5.0)
            cv2.destroyAllWindows()
            
            # Print FPS benchmark stats
            total_time = time.perf_counter() - self._fps_timer
            actual_fps = self.frame_count / total_time if total_time > 0 else 0
            print(f"[BENCHMARK] Average processing FPS: {actual_fps:.2f}")
            print(f"[BENCHMARK] Frames processed: {self.frame_count}/{total}")
            print(f"[BENCHMARK] Reader dropped frames: {reader.dropped_frames}")
            print(f"[BENCHMARK] Display drop rate: {dropped_display_frames/(total_display_frames+dropped_display_frames+1e-5):.1%}")
            avg_gap = self._total_frame_gap / max(self.frame_count, 1)
            print(f"[BENCHMARK] Avg frame gap: {avg_gap:.2f} | Max frame gap: {self._max_frame_gap}")

            # Flush any pending deferred alert before shutdown
            self.alerter.flush_if_ready(
                self.frame_count + self.alerter._alert_window_frames,
                logger=self.logger,
            )

            # Reset all modules
            self.preprocessor.reset()
            self.tracker.reset()
            self.anomaly_det.reset()
            self.classifier.reset()
            self.vehicle_det.reset()
            self.hazard_det.reset()

            print(
                f"[VisionGuard] Done. Total events logged: "
                f"{len(self.logger.events)}"
            )

    # ══════════════════════════════════════════════════════════════
    # DETECTION THREAD
    # ══════════════════════════════════════════════════════════════

    def _detection_loop(self) -> None:
        """
        Detection thread loop. Processes frames from frame_queue
        and puts annotated results into display_queue.
        """
        logger.info("Detection loop started")

        while not self._stop_event.is_set():
            # ── Get (source_frame_num, frame) with timeout ────────
            try:
                item = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Unpack tuple from FrameReader
            if isinstance(item, tuple) and len(item) == 2:
                source_frame_num, frame = item
            else:
                # Backwards compatibility: bare frame
                frame = item
                source_frame_num = self._last_source_frame + 1

            # Track frame gaps
            gap = source_frame_num - self._last_source_frame
            self._total_frame_gap += gap
            self._max_frame_gap = max(self._max_frame_gap, gap)
            if gap > 1:
                logger.debug(
                    "Frame gap detected: source %d -> %d (gap=%d)",
                    self._last_source_frame, source_frame_num, gap,
                )
            self._last_source_frame = source_frame_num

            self.frame_count += 1

            try:
                annotated = self._process_frame(frame, source_frame_num)
            except Exception:
                logger.exception("Error processing frame %d", self.frame_count)
                annotated = frame

            # Non-blocking put into display queue
            try:
                self._display_queue.put_nowait(annotated)
            except queue.Full:
                # Drop oldest, put new
                try:
                    self._display_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._display_queue.put_nowait(annotated)
                except queue.Full:
                    pass

        logger.info("Detection loop exited")

    def _process_frame(self, frame: np.ndarray,
                       source_frame_num: int = None) -> np.ndarray:
        """
        Full pipeline processing for a single frame.

        Stage 1: Adaptive preprocessing + motion gate
        Stage 2: YOLO-A vehicle detection (interval-gated)
        Stage 3: ByteTrack + LK optical flow tracking
        Stage 4: Anomaly detection
        Stage 5: YOLO-B hazard detection (anomaly-gated)
        Stage 6: Severity classification + logging
        Stage 7: Annotation
        """
        # ── Stage 1: Preprocessing + motion gate ──────────────────
        processed, motion_result = self.preprocessor.process(frame)

        if not motion_result.has_motion:
            if self._cached_display_frame is not None:
                return self._cached_display_frame
            return frame

        # ── Stage 2: YOLO-A (interval-gated) ─────────────────────
        if self.frame_count % self._detect_interval == 0:
            vehicle_dets = self.vehicle_det.detect(
                processed, frame_num=self.frame_count, force=True
            )
            self._cached_vehicle_dets = vehicle_dets
        else:
            vehicle_dets = self._cached_vehicle_dets

        # ── Stage 3: ByteTrack + LK optical flow ─────────────────
        tracks = self.tracker.update(processed, vehicle_dets,
                                     source_frame_num=source_frame_num)

        # ── Stage 4: Anomaly detection ────────────────────────────
        anomalies = self.anomaly_det.detect(tracks, self.frame_count)

        # ── Stage 5: YOLO-B (anomaly-gated) ──────────────────────
        hazard_dets = self.hazard_det.detect(
            processed, anomalies, self.frame_count
        )

        # ── Stage 6: Buffer events (defer logging to alert window) ──
        for anomaly in anomalies:
            event = self.classifier.classify(anomaly, hazard_dets)
            severity = event.get("severity", "Minor")
            all_signals = event.get("all_signals", {})
            signal_count = len(all_signals)
            self._active_events[anomaly["track_id"]] = severity

            # Require minimum number of signals to avoid false positives
            if signal_count < self._min_signals:
                continue

            if severity == "Minor" and not self._log_minor:
                continue

            # Capture annotated frame with this event (for snapshot later)
            annotated_for_log = self._annotate(
                frame.copy(), tracks, hazard_dets, anomalies
            )
            # Buffer in alerter — do NOT log yet
            self.alerter.send(
                event, current_frame=self.frame_count,
                annotated_frame=annotated_for_log,
            )

            print(
                f"[EVENT] Frame {self.frame_count} | "
                f"{severity} | Score: {event['total_score']} | "
                f"Signals ({signal_count}): {list(all_signals.keys())}"
            )

        # ── Flush: log + alert ONLY the peak event ─────────────────
        self.alerter.flush_if_ready(
            self.frame_count, logger=self.logger
        )

        # ── Stage 7: Annotate for display ─────────────────────────
        annotated = self._annotate(frame, tracks, hazard_dets, anomalies)
        return annotated

    # ══════════════════════════════════════════════════════════════
    # ANNOTATION
    # ══════════════════════════════════════════════════════════════

    def _annotate(
        self,
        frame: np.ndarray,
        tracks: dict,
        hazard_dets: list,
        anomalies: list,
    ) -> np.ndarray:
        """Draw tracks, hazards, and HUD overlay onto frame."""
        out = frame.copy()

        # ── Draw tracks ──────────────────────────────────────────
        anomaly_tids = {a["track_id"] for a in anomalies}

        for tid, track in tracks.items():
            x1, y1, x2, y2 = track["bbox"]
            severity = self._active_events.get(tid)
            color = self.SEVERITY_COLORS.get(severity, (0, 255, 0))

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label with filled background
            label = f"#{tid} {track['class_name']} {track['speed']:.1f}px/f"
            if severity:
                label += f" [{severity}]"

            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                out,
                (x1, y1 - label_size[1] - baseline - 4),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            # Velocity arrow (scaled 8x)
            vx, vy = track["velocity"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            end_x = int(cx + vx * 8)
            end_y = int(cy + vy * 8)
            cv2.arrowedLine(out, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)

        # ── Draw hazards ─────────────────────────────────────────
        for det in hazard_dets:
            hx1, hy1, hx2, hy2 = det["bbox"]
            name = det["class_name"]
            conf = det["confidence"]
            color = self.HAZARD_COLORS.get(name, (255, 255, 255))

            cv2.rectangle(out, (hx1, hy1), (hx2, hy2), color, 2)
            cv2.putText(
                out,
                f"{name} {conf:.2f}",
                (hx1, hy2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # ── HUD overlay ─────────────────────────────────────────
        self._draw_hud(out, len(tracks), len(hazard_dets), len(anomalies))

        return out

    def _draw_hud(
        self,
        frame: np.ndarray,
        n_vehicles: int,
        n_hazards: int,
        n_anomalies: int,
    ) -> None:
        """Draw semi-transparent HUD with stats."""
        # Compute FPS
        now = time.perf_counter()
        elapsed = now - self._fps_timer
        if elapsed > 0:
            self._fps_counter = self.frame_count / elapsed

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        y = 20
        for text in [
            f"Frame: {self.frame_count}",
            f"FPS: {self._fps_counter:.1f}",
            f"Vehicles: {n_vehicles}",
            f"Hazards: {n_hazards}",
        ]:
            cv2.putText(
                frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            y += 20

        if n_anomalies > 0:
            cv2.putText(
                frame,
                f"ANOMALIES: {n_anomalies}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Synchronous single-frame processing for Streamlit dashboard.

        This runs the full pipeline on a single frame WITHOUT threading.
        Used by dashboard/app.py which needs per-frame results.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame.

        Returns
        -------
        tuple[np.ndarray, list]
            (annotated_frame, list_of_classified_events)
        """
        self.frame_count += 1

        # Stage 1: Preprocessing + motion gate
        processed, motion_result = self.preprocessor.process(frame)

        if not motion_result.has_motion:
            out = frame.copy()
            if hasattr(self, "_last_tracks"):
                out = self._annotate(out, self._last_tracks, self._last_hazards, [])
            return out, []

        # Stage 2: YOLO-A
        if self.frame_count % self._detect_interval == 0:
            vehicle_dets = self.vehicle_det.detect(
                processed, frame_num=self.frame_count, force=True
            )
            self._cached_vehicle_dets = vehicle_dets
        else:
            vehicle_dets = self._cached_vehicle_dets

        # Stage 3: Tracking
        tracks = self.tracker.update(processed, vehicle_dets)

        # Stage 4: Anomaly detection
        anomalies = self.anomaly_det.detect(tracks, self.frame_count)

        # Stage 5: YOLO-B
        hazard_dets = self.hazard_det.detect(
            processed, anomalies, self.frame_count
        )

        # Stage 6: Buffer events (defer logging to alert window)
        classified_events = []
        for anomaly in anomalies:
            event = self.classifier.classify(anomaly, hazard_dets)
            severity = event.get("severity", "Minor")
            all_signals = event.get("all_signals", {})
            signal_count = len(all_signals)
            self._active_events[anomaly["track_id"]] = severity

            if severity == "Minor":
                continue

            if signal_count < self._min_signals:
                continue

            classified_events.append(event)
            annotated_for_log = self._annotate(
                frame.copy(), tracks, hazard_dets, anomalies
            )
            # Buffer — do NOT log yet
            self.alerter.send(
                event, current_frame=self.frame_count,
                annotated_frame=annotated_for_log,
            )

        # Flush: log + alert ONLY the peak event
        self.alerter.flush_if_ready(
            self.frame_count, logger=self.logger
        )

        # Cache for static frames
        self._last_tracks = tracks
        self._last_hazards = hazard_dets

        # Stage 7: Annotate
        out = self._annotate(frame.copy(), tracks, hazard_dets, anomalies)
        return out, classified_events

    def reset(self) -> None:
        """Reset the engine and all sub-modules."""
        self._stop_event.set()
        self.preprocessor.reset()
        self.tracker.reset()
        self.anomaly_det.reset()
        self.classifier.reset()
        self.vehicle_det.reset()
        self.hazard_det.reset()
        self.frame_count = 0
        self._cached_display_frame = None
        self._cached_vehicle_dets = []
        self._active_events.clear()
        logger.info("VisionGuardEngine reset")

    def __repr__(self) -> str:
        return (
            f"VisionGuardEngine(interval={self._detect_interval}, "
            f"queue={self._queue_size})"
        )
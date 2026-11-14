import requests
import time
import os
from datetime import datetime


class TelegramAlert:
    """
    Sends accident alerts to Telegram with snapshot image.
    Respects cooldown to avoid spamming.
    """

    SEVERITY_ORDER = {"Minor": 0, "Moderate": 1, "Severe": 2, "Critical": 3}

    SEVERITY_EMOJI = {
        "Minor"   : "🟡",
        "Moderate": "🟠",
        "Severe"  : "🔴",
        "Critical": "🚨"
    }

    def __init__(self, config: dict):
        acfg = config.get("alerts", {})

        self.enabled     = acfg.get("enable_telegram", False)
        self.bot_token   = acfg.get("bot_token", "")
        self.chat_id     = acfg.get("chat_id", "")
        self.min_severity = acfg.get("min_severity_to_alert", "Moderate")
        self.cooldown    = acfg.get("cooldown_seconds", 30)

        # Optional location info from config
        self._location_name = acfg.get("camera_location", "")
        self._gps_lat = acfg.get("gps_latitude", None)
        self._gps_lon = acfg.get("gps_longitude", None)

        # Deferred alert: buffer events and send the peak-score frame
        self._alert_window_frames = acfg.get("alert_window_frames", 30)

        self._last_sent  = 0   # timestamp of last alert sent
        self._base_url   = f"https://api.telegram.org/bot{self.bot_token}"

        # ── Peak-frame buffering state ─────────────────────────────
        self._pending_events = []       # list of (event_dict) during window
        self._window_start_frame = None # frame_number when window opened
        self._incident_vehicles = {}    # track_id -> class_name (only involved)

    def _should_send(self, severity: str) -> bool:
        """Check severity threshold and cooldown."""
        if not self.enabled:
            return False
        if not self.bot_token or not self.chat_id:
            return False

        # check severity meets minimum threshold
        event_level = self.SEVERITY_ORDER.get(severity, 0)
        min_level   = self.SEVERITY_ORDER.get(self.min_severity, 1)
        if event_level < min_level:
            return False

        # check cooldown
        if time.time() - self._last_sent < self.cooldown:
            return False

        return True

    def _build_message(self, event: dict) -> str:
        severity  = event.get("severity", "Unknown")
        emoji     = self.SEVERITY_EMOJI.get(severity, "⚠️")
        track_id  = event.get("track_id", "N/A")
        vehicle   = event.get("class_name", "unknown")
        score     = event.get("total_score", 0)
        signals   = list(event.get("signals", {}).keys())
        frame     = event.get("frame_number", "N/A")
        speed     = event.get("speed", 0)

        # Track vehicles involved in this incident
        self._incident_vehicles[track_id] = vehicle

        # Format date and time
        timestamp = event.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp)
            date_str = dt.strftime("%d %B %Y")
            time_str = dt.strftime("%I:%M:%S %p")
        except Exception:
            date_str = "Unknown"
            time_str = timestamp

        # Count and list vehicles involved
        num_vehicles = len(self._incident_vehicles)
        vehicle_types = list(set(self._incident_vehicles.values()))
        vehicle_list = ", ".join(
            f"{vehicle_types.count(v)}x {v}" for v in set(vehicle_types)
        )

        # Build signal descriptions in plain English
        signal_descriptions = []
        signal_map = {
            "speed_drop_heavy": "Sudden vehicle stop detected",
            "speed_drop_moderate": "Significant deceleration",
            "trajectory_deviation": "Abnormal direction change",
            "flow_incoherence": "Chaotic motion pattern",
            "vehicle_overlap": "Vehicle collision / overlap",
            "crashed_vehicle_yolo": "Crash damage visible (AI)",
            "fire_detected": "🔥 Fire detected near scene",
            "smoke_detected": "💨 Smoke detected near scene",
            "motorcycle_track_lost": "Motorcycle disappeared suddenly",
        }
        for sig in signals:
            desc = signal_map.get(sig, sig.replace("_", " ").title())
            signal_descriptions.append(f"  • {desc}")
        signals_text = "\n".join(signal_descriptions) if signal_descriptions else "  • None"

        # Location block
        location_block = ""
        if self._location_name:
            location_block += f"📍 *Location:* {self._location_name}\n"
        if self._gps_lat is not None and self._gps_lon is not None:
            location_block += f"🌐 *GPS:* {self._gps_lat}, {self._gps_lon}\n"

        msg = (
            f"{emoji} *VISIONGUARD — INCIDENT ALERT* {emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"🔺 *Severity:* {severity}\n"
            f"📊 *Risk Score:* {score}\n"
            f"\n"
            f"🚗 *Vehicles Involved:* {num_vehicles}\n"
            f"🏷️ *Vehicle Types:* {vehicle_list}\n"
            f"\n"
            f"📅 *Date:* {date_str}\n"
            f"🕐 *Time:* {time_str}\n"
            f"{location_block}"
            f"\n"
            f"⚡ *Detected Signals:*\n"
            f"{signals_text}\n"
            f"\n"
            f"🎞 *Frame:* {frame}\n"
            f"💨 *Speed:* {speed:.1f} px/frame\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"_Automated alert by VisionGuard AI_"
        )
        return msg

    def _send_text(self, message: str) -> bool:
        try:
            url  = f"{self._base_url}/sendMessage"
            data = {
                "chat_id"   : self.chat_id,
                "text"      : message,
                "parse_mode": "Markdown"
            }
            resp = requests.post(url, data=data, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"[Telegram] Text send failed: {e}")
            return False

    def _send_photo(self, image_path: str, caption: str) -> bool:
        try:
            url = f"{self._base_url}/sendPhoto"
            with open(image_path, "rb") as img:
                resp = requests.post(
                    url,
                    data={
                        "chat_id"   : self.chat_id,
                        "caption"   : caption,
                        "parse_mode": "Markdown"
                    },
                    files={"photo": img},
                    timeout=15
                )
            return resp.status_code == 200
        except Exception as e:
            print(f"[Telegram] Photo send failed: {e}")
            return False

    def send(self, event: dict, current_frame: int = 0,
             annotated_frame=None):
        """
        Buffer an event + its annotated frame into the alert window.

        Events are NOT logged or alerted immediately. They accumulate
        during the window. When the window closes, the BEST event
        (peak composite score) is logged and alerted.
        """
        severity = event.get("severity", "Minor")
        event_level = self.SEVERITY_ORDER.get(severity, 0)
        min_level   = self.SEVERITY_ORDER.get(self.min_severity, 1)
        if event_level < min_level:
            return

        if not self.enabled or not self.bot_token or not self.chat_id:
            return

        # Track vehicles with anomaly signals as "involved"
        tid = event.get("track_id")
        if tid is not None:
            self._incident_vehicles[tid] = event.get("class_name", "unknown")

        # Open a new window if none is active
        if self._window_start_frame is None:
            self._window_start_frame = current_frame
            print(f"[Telegram] Alert window opened at frame {current_frame}")

        # Store event + frame + frame_number together
        self._pending_events.append({
            "event": event,
            "frame": annotated_frame,
            "frame_number": current_frame,
        })

    def _compute_peak_score(self, entry: dict) -> float:
        """
        Compute a composite score that favors POST-COLLISION frames.

        The raw total_score treats pre-crash and post-crash signals
        equally. But the actual accident frame has:
          - LOW speed (vehicle stopped after impact)
          - HIGH signal count (multiple signals fire simultaneously)
          - YOLO-B crashed_vehicle confirmation

        This composite score adds bonuses for those properties.
        """
        event = entry["event"]
        total_score = event.get("total_score", 0)
        speed = event.get("speed", 999)
        avg_speed = event.get("avg_speed", 1)
        all_signals = event.get("all_signals", {})
        signal_count = len(all_signals)

        # ── Speed drop bonus (up to +5): lower speed = more post-collision ──
        if avg_speed > 0.1:
            speed_ratio = speed / (avg_speed + 1e-6)
            # At speed_ratio=0 (stopped): bonus = 5
            # At speed_ratio=1 (no drop): bonus = 0
            speed_bonus = max(0.0, (1.0 - min(speed_ratio, 1.0)) * 5.0)
        else:
            speed_bonus = 0.0

        # ── Signal count bonus: more signals = stronger evidence ──
        signal_bonus = max(0.0, (signal_count - 1) * 1.0)

        # ── YOLO-B crash confirmation bonus ──
        crash_bonus = 3.0 if "crashed_vehicle_yolo" in all_signals else 0.0

        composite = total_score + speed_bonus + signal_bonus + crash_bonus
        return round(composite, 2)

    def flush_if_ready(self, current_frame: int, logger=None):
        """
        Check if the alert window has expired. If so:
          1. Pick the peak event using composite scoring
          2. Log ONLY that event (via logger)
          3. Send Telegram alert with that event's snapshot

        Parameters
        ----------
        current_frame : int
            Current frame number.
        logger : EventLogger, optional
            Logger to write the peak event. Only the peak is logged.
        """
        if self._window_start_frame is None:
            return  # no active window

        frames_elapsed = current_frame - self._window_start_frame
        if frames_elapsed < self._alert_window_frames:
            return  # window still open, keep collecting

        # ── Window expired — find peak event ──────────────────────
        if not self._pending_events:
            self._reset_window()
            return

        # Check cooldown before sending
        if time.time() - self._last_sent < self.cooldown:
            print(f"[Telegram] Alert window closed but cooldown active, skipping")
            self._reset_window()
            return

        # ── Pick the event with the highest COMPOSITE score ───────
        # This heavily favors post-collision frames (low speed,
        # high signal count, YOLO-B crash confirmation)
        peak_entry = max(
            self._pending_events,
            key=self._compute_peak_score
        )
        peak_event = peak_entry["event"]
        peak_frame = peak_entry["frame"]
        peak_frame_num = peak_entry["frame_number"]
        peak_composite = self._compute_peak_score(peak_entry)
        severity = peak_event.get("severity", "Unknown")

        print(
            f"[Telegram] Window closed: {len(self._pending_events)} events, "
            f"peak at frame {peak_frame_num} "
            f"(composite={peak_composite}, "
            f"raw={peak_event.get('total_score')}, "
            f"speed={peak_event.get('speed', 0):.1f})"
        )

        # ── Log ONLY the peak event ───────────────────────────────
        record = None
        if logger is not None:
            record = logger.log(peak_event, peak_frame, peak_frame_num)

        # ── Send Telegram alert ───────────────────────────────────
        message = self._build_message(peak_event)
        snapshot = record.get("snapshot") if record else None

        # Format date/time for photo caption
        timestamp = peak_event.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp)
            caption_time = dt.strftime("%d %b %Y, %I:%M %p")
        except Exception:
            caption_time = datetime.now().strftime("%d %b %Y, %I:%M %p")

        num_vehicles = len(self._incident_vehicles)

        # Send photo with caption if snapshot exists, else text only
        if snapshot and os.path.exists(snapshot):
            short_caption = (
                f"{self.SEVERITY_EMOJI.get(severity, '⚠️')} "
                f"*{severity} Incident Detected*\n"
                f"🚗 {num_vehicles} vehicle(s) involved\n"
                f"📊 Score: {peak_event.get('total_score', 0)}\n"
                f"🕐 {caption_time}"
            )
            success = self._send_photo(snapshot, short_caption)
            if success:
                self._send_text(message)
        else:
            success = self._send_text(message)

        # Send GPS location pin if coordinates are available
        if success and self._gps_lat is not None and self._gps_lon is not None:
            self._send_location(self._gps_lat, self._gps_lon)

        if success:
            self._last_sent = time.time()
            print(f"[Telegram] Alert sent — {severity} (peak frame {peak_frame_num})")
        else:
            print(f"[Telegram] Alert failed to send")

        self._reset_window()

    def _reset_window(self):
        """Clear the buffering state after an alert is sent or discarded."""
        # Release frame references to free memory
        self._pending_events.clear()
        self._window_start_frame = None
        self._incident_vehicles.clear()

    def _send_location(self, latitude: float, longitude: float) -> bool:
        """Send a GPS location pin to the Telegram chat."""
        try:
            url = f"{self._base_url}/sendLocation"
            data = {
                "chat_id": self.chat_id,
                "latitude": latitude,
                "longitude": longitude,
            }
            resp = requests.post(url, data=data, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"[Telegram] Location send failed: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Call this once at startup to verify bot token and chat ID work.
        """
        try:
            msg = (
                "✅ *VisionGuard Connected*\n"
                "Alert system is active and working."
            )
            return self._send_text(msg)
        except Exception:
            return False
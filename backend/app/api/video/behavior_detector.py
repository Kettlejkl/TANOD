import numpy as np
import cv2
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

try:
    from .yolo_model import register_violence_event, reset_violence_state
    FACE_BLUR = True
except ImportError:
    FACE_BLUR = False


@dataclass
class PerspectiveConfig:
    enabled: bool = True
    vanishing_point_y: float = 0.3
    horizon_line_y: Optional[int] = None
    depth_scaling_power: float = 2.0


class PerspectiveCorrector:

    def __init__(self, camera_config, perspective_config: PerspectiveConfig):
        self.camera = camera_config
        self.perspective = perspective_config

        if not perspective_config.enabled:
            self.enabled = False
            self.base_px_per_meter = camera_config.pixels_per_meter()
            return

        self.enabled = True

        if perspective_config.horizon_line_y is None:
            self.horizon_y = int(camera_config.resolution[1] *
                                 perspective_config.vanishing_point_y)
        else:
            self.horizon_y = perspective_config.horizon_line_y

        self.base_px_per_meter = camera_config.pixels_per_meter()

    def get_depth_factor(self, bbox_or_point) -> float:
        if not self.enabled:
            return 1.0

        if isinstance(bbox_or_point, (list, tuple)):
            if len(bbox_or_point) == 4:
                y = bbox_or_point[1] + bbox_or_point[3]
            else:
                y = bbox_or_point[1]
        else:
            y = bbox_or_point

        frame_height = self.camera.resolution[1]
        distance_from_horizon = max(0, y - self.horizon_y) / max(1, frame_height - self.horizon_y)
        scale = 1.0 + (self.perspective.depth_scaling_power * (1.0 - distance_from_horizon))

        return max(1.0, scale)

    def get_adjusted_px_per_meter(self, bbox_y: float) -> float:
        if not self.enabled:
            return self.base_px_per_meter

        depth_scale = self.get_depth_factor(bbox_y)
        return self.base_px_per_meter / depth_scale

    def pixels_to_meters(self, pixel_distance: float, bbox_y: float) -> float:
        px_per_m = self.get_adjusted_px_per_meter(bbox_y)
        return pixel_distance / px_per_m

    def meters_to_pixels(self, meter_distance: float, bbox_y: float) -> float:
        px_per_m = self.get_adjusted_px_per_meter(bbox_y)
        return meter_distance * px_per_m


@dataclass
class CameraConfig:
    height_meters: float = 3.0
    fov_degrees: float = 90.0
    resolution: Tuple[int, int] = (1920, 1080)
    fps: float = 10.0

    def pixels_per_meter(self) -> float:
        ground_distance = self.height_meters * 1.5
        frame_width_meters = 2 * ground_distance * math.tan(math.radians(self.fov_degrees / 2))
        return self.resolution[0] / frame_width_meters


@dataclass
class ZoneConfig:
    name: str
    zone_type: str
    crowd_threshold: int
    loiter_time: float
    sensitivity: float = 1.0


# --- Fence entry running thresholds ---
RUN_ENTRY_PX_PER_S    = 80.0
RUN_ENTRY_MPS         = 2.8
RUN_ENTRY_MIN_CONF    = 0.45          # FIX: raised from 0.35

# --- Speed-based running thresholds ---
RUN_MPS               = 3.5
SPRINT_MPS            = 5.5
SPRINT_MIN_FRAMES     = 3
RUN_CONFIDENCE_MIN    = 0.55          # FIX: raised from 0.4
SPRINT_CONFIDENCE_MIN = 0.3

# --- Gait-based running thresholds (all raised to reduce false positives) ---
GAIT_SWING_THRESHOLD      = 35        # FIX: raised from 20 — 20px is noise / brisk walk
GAIT_HIP_THRESHOLD        = 12        # FIX: raised from 5 — 5px is noise
GAIT_KNEE_DRIVE_THRESHOLD = 0.5       # FIX: raised from 0.3 — majority of frames must show drive
GAIT_MIN_SIGNALS          = 3         # FIX: raised from 2 — require ALL signals (stricter)

# --- Warmup: minimum velocity samples before any running alert fires ---
RUN_MIN_VEL_HISTORY       = 8         # FIX: new constant

# --- Consistency: fraction of recent frames that must exceed the run threshold ---
RUN_CONSISTENCY_MIN       = 0.8       # FIX: raised from 0.6 (4/5 frames instead of 3/5)

# --- Re-alert cooldown per PID per behavior (seconds) ---
BEHAVIOR_COOLDOWN         = 30.0      # FIX: new constant — prevents re-alert storm


class AdaptiveBehaviorDetector:

    def __init__(self, camera_config: Optional[CameraConfig] = None,
                 zones: Optional[List[ZoneConfig]] = None):

        self.camera = camera_config or CameraConfig()
        self.zones = {z.name: z for z in (zones or [])}

        perspective_config = PerspectiveConfig(
            enabled=True,
            vanishing_point_y=0.15,
            depth_scaling_power=1.2
        )
        self.perspective = PerspectiveCorrector(self.camera, perspective_config)

        px_per_meter = self.camera.pixels_per_meter()

        walking_speed = 1.4
        running_speed = 3.5
        frame_time    = 1.0 / self.camera.fps

        self.WALK_SPEED = walking_speed * px_per_meter * frame_time
        self.RUN_SPEED  = running_speed * px_per_meter * frame_time

        self.VIOLENCE_SPEED_LOW  = 4.0 * px_per_meter * frame_time
        self.VIOLENCE_SPEED_HIGH = 8.0 * px_per_meter * frame_time

        self.LOITER_TIME_BASE = 300.0
        self.LOITER_DIST      = 1.5 * px_per_meter
        self.CROWD_COUNT_BASE = 15

        self.MIN_RUNNING_FRAMES  = 5
        self.MIN_VIOLENCE_FRAMES = 4

        self.MAX_REASONABLE_VELOCITY = 10.0 * px_per_meter * frame_time
        self.MAX_ARM_VELOCITY        = 12.0 * px_per_meter * frame_time

        self.tracks = defaultdict(lambda: {
            'pos':                    deque(maxlen=50),
            'ts':                     deque(maxlen=50),
            'vel':                    deque(maxlen=25),
            'kp':                     deque(maxlen=25),
            'aspect':                 deque(maxlen=30),
            'first_seen':             None,
            'last_seen':              None,
            'violent':                False,
            'alerted_behaviors':      set(),
            'alerted_behavior_ts':    {},   # FIX: tracks timestamp of last alert per behavior
            'zone':                   None,
            'behavior_stats':         defaultdict(int),
            'fence_entry_speed_px_s': None,
            'fence_entry_speed_mps':  None,
            'fence_entry_pos':        None,
            'fence_entry_ts':         None,
            'fence_entry_name':       None,
            'fence_entry_id':         None,
        })

        self.frame_count      = 0
        self.last_alerts      = defaultdict(float)
        self.velocity_history = deque(maxlen=1000)
        self.crowd_history    = deque(maxlen=100)
        self.baseline_frames  = 0
        self.ts_now           = 0.0   # FIX: current frame timestamp, set in analyze_frame

    def _get_zone_for_position(self, pos: Tuple[int, int]) -> Optional[ZoneConfig]:
        return list(self.zones.values())[0] if self.zones else None

    def _get_adjusted_threshold(self, base_value: float, zone: Optional[ZoneConfig]) -> float:
        if zone:
            return base_value * zone.sensitivity
        return base_value

    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _velocity(self, p1, p2, dt):
        if dt <= 0:
            return 0.0
        return min(self._dist(p1, p2) / dt, self.MAX_REASONABLE_VELOCITY)

    def _arm_speed(self, kp1, kp2, dt):
        if not kp1 or not kp2 or dt <= 0:
            return 0.0
        try:
            dists = []
            for k in ['left_wrist', 'right_wrist', 'left_elbow', 'right_elbow']:
                if k in kp1 and k in kp2:
                    dists.append(self._dist(kp1[k], kp2[k]))
            if not dists:
                return 0.0
            return min(max(dists) / dt, self.MAX_ARM_VELOCITY)
        except Exception:
            return 0.0

    def _extract_keypoints(self, pose_results, bbox):
        if not pose_results or len(pose_results) == 0:
            return None
        try:
            bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            best_iou, best_idx = 0.3, None

            for idx, pose_box in enumerate(pose_results[0].boxes):
                pb = pose_box.xyxy[0].cpu().numpy()
                x1 = max(bbox_xyxy[0], pb[0])
                y1 = max(bbox_xyxy[1], pb[1])
                x2 = min(bbox_xyxy[2], pb[2])
                y2 = min(bbox_xyxy[3], pb[3])

                if x2 > x1 and y2 > y1:
                    inter = (x2 - x1) * (y2 - y1)
                    a1    = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
                    a2    = (pb[2] - pb[0]) * (pb[3] - pb[1])
                    iou   = inter / (a1 + a2 - inter + 1e-6)
                    if iou > best_iou:
                        best_iou, best_idx = iou, idx

            if best_idx is not None:
                kp = pose_results[0].keypoints.xy.cpu().numpy()[best_idx]
                return {k: kp[i].tolist() for k, i in [
                    ('left_wrist',     9),
                    ('right_wrist',   10),
                    ('left_elbow',     7),
                    ('right_elbow',    8),
                    ('left_shoulder',  5),
                    ('right_shoulder', 6),
                    ('left_hip',      11),
                    ('right_hip',     12)
                ]}
        except Exception:
            pass
        return None

    def _can_alert_behavior(self, pid, behavior_type):
        """
        FIX: Two-tier guard.
          1. Once-per-track block (alerted_behaviors set) — keeps original semantics.
          2. Time-based cooldown (alerted_behavior_ts) — prevents re-alert storms
             if the set is ever cleared externally.
        """
        t = self.tracks[pid]
        if behavior_type in t['alerted_behaviors']:
            return False
        last_ts = t['alerted_behavior_ts'].get(behavior_type, 0.0)
        if (self.ts_now - last_ts) < BEHAVIOR_COOLDOWN:
            return False
        return True

    def _mark_behavior_alerted(self, pid, behavior_type):
        t = self.tracks[pid]
        t['alerted_behaviors'].add(behavior_type)
        t['alerted_behavior_ts'][behavior_type] = self.ts_now   # FIX: record timestamp
        t['behavior_stats'][behavior_type] += 1

    def _can_alert_zone(self, key, ts, cooldown=60.0):
        return ts - self.last_alerts.get(key, 0) >= cooldown

    def _calculate_confidence(self, metric, threshold, max_value):
        if metric < threshold:
            return 0.0
        excess = min(metric - threshold, max_value - threshold)
        return min(excess / (max_value - threshold), 1.0)

    # ------------------------------------------------------------------
    # Fence-entry helpers
    # ------------------------------------------------------------------

    def record_fence_entry(self, pid, speed_px_s: float, centroid: tuple,
                           ts: float, fence_name: str, fence_id: str,
                           bbox_y: float = 0.0):
        t = self.tracks[pid]

        speed_mps = None
        if self.perspective.enabled and speed_px_s > 0:
            speed_mps = round(
                self.perspective.pixels_to_meters(speed_px_s, bbox_y), 2)

        t['fence_entry_speed_px_s'] = speed_px_s
        t['fence_entry_speed_mps']  = speed_mps
        t['fence_entry_pos']        = centroid
        t['fence_entry_ts']         = ts
        t['fence_entry_name']       = fence_name
        t['fence_entry_id']         = fence_id

        print(f"[BehaviorDetector] Entry recorded for {pid}: "
              f"{speed_px_s:.0f} px/s"
              + (f" / {speed_mps:.2f} m/s" if speed_mps is not None else "")
              + f" -> fence '{fence_name}'")

    def clear_fence_entry(self, pid):
        t = self.tracks[pid]
        t['fence_entry_speed_px_s'] = None
        t['fence_entry_speed_mps']  = None
        t['fence_entry_pos']        = None
        t['fence_entry_ts']         = None
        t['fence_entry_name']       = None
        t['fence_entry_id']         = None

    # ------------------------------------------------------------------
    # Detection: fence entry running
    # ------------------------------------------------------------------

    def detect_fence_entry_running(self, pid, ts):
        t = self.tracks[pid]

        if t['fence_entry_speed_px_s'] is None:
            return None
        if not self._can_alert_behavior(pid, 'fence_entry_running'):
            return None

        speed_px_s = t['fence_entry_speed_px_s']
        speed_mps  = t['fence_entry_speed_mps']
        fence_name = t['fence_entry_name'] or 'unknown'
        fence_id   = t['fence_entry_id']
        pos        = t['fence_entry_pos'] or (0, 0)
        bbox_y     = pos[1]

        if speed_mps is not None and self.perspective.enabled:
            is_running  = speed_mps >= RUN_ENTRY_MPS
            speed_label = f"{speed_mps:.2f} m/s"
            max_speed   = RUN_ENTRY_MPS * 3.0
            confidence  = self._calculate_confidence(speed_mps, RUN_ENTRY_MPS, max_speed)
        else:
            is_running  = speed_px_s >= RUN_ENTRY_PX_PER_S
            speed_label = f"{speed_px_s:.0f} px/s"
            max_speed   = RUN_ENTRY_PX_PER_S * 3.0
            confidence  = self._calculate_confidence(speed_px_s, RUN_ENTRY_PX_PER_S, max_speed)

        if not is_running or confidence < RUN_ENTRY_MIN_CONF:
            return None

        self._mark_behavior_alerted(pid, 'fence_entry_running')
        self._mark_behavior_alerted(pid, 'running')

        result = {
            'type':       'running',
            'subtype':    'fence_entry',
            'person_id':  pid,
            'position':   [int(pos[0]), int(pos[1])],
            'confidence': round(confidence, 2),
            'fence_name': fence_name,
            'fence_id':   fence_id,
            'zone':       t['zone'].name if t['zone'] else fence_name,
        }

        if speed_mps is not None:
            result['speed_mps']    = speed_mps
            result['speed_kmh']    = round(speed_mps * 3.6, 2)
            result['depth_factor'] = round(self.perspective.get_depth_factor(bbox_y), 2)
        else:
            result['speed_px_s'] = round(speed_px_s, 1)

        print(f"[RUNNING-ENTRY] UID {pid} entered '{fence_name}' "
              f"at {speed_label}  conf={confidence:.2f}")

        return result

    # ------------------------------------------------------------------
    # Track update
    # ------------------------------------------------------------------

    def update_person(self, pid, bbox, ts, keypoints=None):
        t = self.tracks[pid]
        if t['first_seen'] is None:
            t['first_seen'] = ts
        t['last_seen'] = ts

        centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        t['pos'].append(centroid)
        t['ts'].append(ts)

        t['aspect'].append(bbox[3] / max(bbox[2], 1))
        t['zone'] = self._get_zone_for_position(centroid)

        if keypoints:
            t['kp'].append({'kp': keypoints, 'ts': ts})

        if len(t['pos']) >= 2:
            dt = t['ts'][-1] - t['ts'][-2]
            if dt > 0:
                v = self._velocity(t['pos'][-1], t['pos'][-2], dt)
                if v > 0:
                    t['vel'].append(v)
                    self.velocity_history.append(v)

    # ------------------------------------------------------------------
    # Detection: gait-based running
    # ------------------------------------------------------------------

    def detect_running_by_gait(self, pid, ts):
        t = self.tracks[pid]
        if len(t['kp']) < 6:
            return None
        if not self._can_alert_behavior(pid, 'running'):
            return None

        # FIX: velocity pre-check — gait alone must not fire on slow movers.
        # Require at least RUN_MIN_VEL_HISTORY samples and a minimum speed
        # above a brisk walk before even evaluating gait signals.
        if len(t['vel']) < RUN_MIN_VEL_HISTORY:
            return None
        avg_v = np.median(list(t['vel'])[-RUN_MIN_VEL_HISTORY:])
        if avg_v < self.WALK_SPEED * 1.5:
            return None

        recent_kp = list(t['kp'])[-6:]

        left_wrist_ys  = []
        right_wrist_ys = []

        for frame_kp in recent_kp:
            kp = frame_kp['kp']
            if kp and 'left_wrist' in kp and 'right_wrist' in kp:
                left_wrist_ys.append(kp['left_wrist'][1])
                right_wrist_ys.append(kp['right_wrist'][1])

        if len(left_wrist_ys) < 4:
            return None

        left_swing  = max(left_wrist_ys) - min(left_wrist_ys)
        right_swing = max(right_wrist_ys) - min(right_wrist_ys)
        avg_swing   = (left_swing + right_swing) / 2

        hip_ys = []
        for frame_kp in recent_kp:
            kp = frame_kp['kp']
            if kp and 'left_hip' in kp and 'right_hip' in kp:
                mid_hip_y = (kp['left_hip'][1] + kp['right_hip'][1]) / 2
                hip_ys.append(mid_hip_y)

        hip_oscillation = (max(hip_ys) - min(hip_ys)) if len(hip_ys) >= 4 else 0

        knee_drive_score = 0
        for frame_kp in recent_kp:
            kp = frame_kp['kp']
            if not kp:
                continue
            if 'left_wrist' in kp and 'left_hip' in kp:
                if kp['left_wrist'][1] < kp['left_hip'][1]:
                    knee_drive_score += 1
            if 'right_wrist' in kp and 'right_hip' in kp:
                if kp['right_wrist'][1] < kp['right_hip'][1]:
                    knee_drive_score += 1

        knee_drive_ratio = knee_drive_score / (len(recent_kp) * 2)

        # FIX: all three signals must pass (GAIT_MIN_SIGNALS = 3),
        # and each threshold is substantially higher than before.
        score = 0
        if avg_swing        > GAIT_SWING_THRESHOLD:      score += 1
        if hip_oscillation  > GAIT_HIP_THRESHOLD:        score += 1
        if knee_drive_ratio > GAIT_KNEE_DRIVE_THRESHOLD: score += 1

        if score >= GAIT_MIN_SIGNALS:
            confidence = score / 3.0
            self._mark_behavior_alerted(pid, 'running')
            return {
                'type':            'running',
                'subtype':         'gait_detected',
                'person_id':       pid,
                'position':        [int(t['pos'][-1][0]), int(t['pos'][-1][1])],
                'confidence':      round(confidence, 2),
                'arm_swing_px':    round(avg_swing, 1),
                'hip_oscillation': round(hip_oscillation, 1),
                'knee_drive':      round(knee_drive_ratio, 2),
                'zone':            t['zone'].name if t['zone'] else 'unknown',
            }

        return None

    # ------------------------------------------------------------------
    # Detection: loitering
    # ------------------------------------------------------------------

    def detect_loitering(self, pid, ts):
        t = self.tracks[pid]
        if not self._can_alert_behavior(pid, 'loitering') or len(t['pos']) < 30:
            return None

        duration         = ts - t['first_seen']
        loiter_threshold = self._get_adjusted_threshold(self.LOITER_TIME_BASE, t['zone'])

        if duration < loiter_threshold:
            return None

        pos         = list(t['pos'])
        x_range     = max(p[0] for p in pos) - min(p[0] for p in pos)
        y_range     = max(p[1] for p in pos) - min(p[1] for p in pos)
        movement_px = max(x_range, y_range)
        bbox_y      = pos[-1][1]

        loiter_threshold_px = self.perspective.meters_to_pixels(1.5, bbox_y)

        if movement_px < loiter_threshold_px:
            avg_v = np.mean(list(t['vel'])[-20:]) if len(t['vel']) >= 20 else float('inf')

            if avg_v < self.WALK_SPEED * 0.3:
                confidence  = self._calculate_confidence(duration, loiter_threshold, loiter_threshold * 2)
                confidence *= (1.0 - min(movement_px / loiter_threshold_px, 1.0))

                if confidence > 0.5:
                    self._mark_behavior_alerted(pid, 'loitering')
                    result = {
                        'type':       'loitering',
                        'person_id':  pid,
                        'duration':   round(duration, 1),
                        'position':   [int(pos[-1][0]), int(pos[-1][1])],
                        'confidence': round(confidence, 2),
                        'zone':       t['zone'].name if t['zone'] else 'unknown'
                    }
                    if self.perspective.enabled:
                        result['movement_meters'] = round(
                            self.perspective.pixels_to_meters(movement_px, bbox_y), 2)
                        result['depth_factor'] = round(
                            self.perspective.get_depth_factor(bbox_y), 2)
                    return result
        return None

    # ------------------------------------------------------------------
    # Detection: speed-based running
    # ------------------------------------------------------------------

    def detect_running(self, pid, ts):
        t = self.tracks[pid]
        if not self._can_alert_behavior(pid, 'running') or len(t['vel']) < self.MIN_RUNNING_FRAMES:
            return None

        # FIX: require a minimum velocity history before firing
        if len(t['vel']) < RUN_MIN_VEL_HISTORY:
            return None

        recent = list(t['vel'])[-self.MIN_RUNNING_FRAMES:]
        bbox_y = t['pos'][-1][1]

        if self.perspective.enabled:
            real_speeds   = [self.perspective.pixels_to_meters(v, bbox_y) for v in recent]
            avg_speed     = np.median(real_speeds)          # FIX: median instead of mean
            run_threshold = RUN_MPS
            fast_frames   = sum(1 for v in real_speeds if v > run_threshold)
        else:
            avg_speed     = np.median(recent)               # FIX: median instead of mean
            run_threshold = self.RUN_SPEED
            fast_frames   = sum(1 for v in recent if v > run_threshold)

        consistency = fast_frames / len(recent)

        # FIX: consistency raised from 0.6 to RUN_CONSISTENCY_MIN (0.8)
        if avg_speed > run_threshold and consistency >= RUN_CONSISTENCY_MIN:
            confidence  = min((avg_speed - run_threshold) / run_threshold, 1.0)
            confidence *= consistency

            # FIX: confidence floor raised from 0.4 to 0.55
            if confidence > RUN_CONFIDENCE_MIN:
                self._mark_behavior_alerted(pid, 'running')
                result = {
                    'type':       'running',
                    'person_id':  pid,
                    'position':   [int(t['pos'][-1][0]), int(t['pos'][-1][1])],
                    'confidence': round(confidence, 2),
                    'zone':       t['zone'].name if t['zone'] else 'unknown'
                }
                if self.perspective.enabled:
                    result['speed_mps']    = round(avg_speed, 2)
                    result['speed_kmh']    = round(avg_speed * 3.6, 2)
                    result['depth_factor'] = round(
                        self.perspective.get_depth_factor(bbox_y), 2)
                else:
                    result['speed'] = round(avg_speed, 1)
                return result

        return None

    # ------------------------------------------------------------------
    # Detection: sprinting
    # ------------------------------------------------------------------

    def detect_sprinting(self, pid, ts):
        t = self.tracks[pid]
        if not self._can_alert_behavior(pid, 'sprinting') or len(t['vel']) < SPRINT_MIN_FRAMES:
            return None

        # FIX: require minimum velocity history
        if len(t['vel']) < RUN_MIN_VEL_HISTORY:
            return None

        recent = list(t['vel'])[-SPRINT_MIN_FRAMES:]
        bbox_y = t['pos'][-1][1]

        if self.perspective.enabled:
            real_speeds   = [self.perspective.pixels_to_meters(v, bbox_y) for v in recent]
            avg_speed     = np.median(real_speeds)          # FIX: median instead of mean
            sprint_thresh = SPRINT_MPS
            fast_frames   = sum(1 for v in real_speeds if v > sprint_thresh)
        else:
            sprint_thresh = SPRINT_MPS * self.camera.pixels_per_meter()
            avg_speed     = np.median(recent)               # FIX: median instead of mean
            fast_frames   = sum(1 for v in recent if v > sprint_thresh)

        consistency = fast_frames / len(recent)

        if avg_speed > sprint_thresh and consistency >= 0.6:
            confidence  = min((avg_speed - sprint_thresh) / sprint_thresh, 1.0)
            confidence *= consistency

            if confidence > SPRINT_CONFIDENCE_MIN:
                self._mark_behavior_alerted(pid, 'sprinting')
                self._mark_behavior_alerted(pid, 'running')

                result = {
                    'type':       'sprinting',
                    'person_id':  pid,
                    'position':   [int(t['pos'][-1][0]), int(t['pos'][-1][1])],
                    'confidence': round(confidence, 2),
                    'zone':       t['zone'].name if t['zone'] else 'unknown',
                }
                if self.perspective.enabled:
                    result['speed_mps']    = round(avg_speed, 2)
                    result['speed_kmh']    = round(avg_speed * 3.6, 2)
                    result['depth_factor'] = round(
                        self.perspective.get_depth_factor(bbox_y), 2)
                else:
                    result['speed'] = round(avg_speed, 1)

                print(f"[SPRINTING] UID {pid} @ "
                      + (f"{avg_speed:.2f} m/s ({avg_speed*3.6:.1f} km/h)"
                         if self.perspective.enabled
                         else f"{avg_speed:.0f} px/s")
                      + f"  conf={confidence:.2f}")

                return result

        return None

    # ------------------------------------------------------------------
    # Detection: violence
    # ------------------------------------------------------------------

    def detect_violence(self, pid, ts, all_pos, camera_id=None):
        t = self.tracks[pid]
        if not self._can_alert_behavior(pid, 'violence') or len(t['kp']) < self.MIN_VIOLENCE_FRAMES:
            return None

        curr_pos            = t['pos'][-1]
        bbox_y              = curr_pos[1]
        proximity_threshold = self.perspective.meters_to_pixels(2.0, bbox_y)

        nearby = {oid: opos for oid, opos in all_pos.items()
                  if oid != pid and self._dist(curr_pos, opos) < proximity_threshold}

        if not nearby:
            return None

        recent_kp  = list(t['kp'])[-self.MIN_VIOLENCE_FRAMES:]
        arm_speeds = [
            self._arm_speed(
                recent_kp[i]['kp'], recent_kp[i - 1]['kp'],
                recent_kp[i]['ts'] - recent_kp[i - 1]['ts']
            )
            for i in range(1, len(recent_kp))
        ]

        if not arm_speeds:
            return None

        max_arm_speed   = max(arm_speeds)
        fast_arm_frames = sum(1 for s in arm_speeds if s > self.VIOLENCE_SPEED_LOW)

        if max_arm_speed > self.VIOLENCE_SPEED_LOW and fast_arm_frames >= 4:
            involved_ids  = [pid]
            violence_type = 'mutual_altercation'
            confidence    = 0.0

            for nid, npos in nearby.items():
                nt = self.tracks[nid]
                if len(nt['kp']) >= 6:
                    nkp     = list(nt['kp'])[-6:]
                    nspeeds = [
                        self._arm_speed(
                            nkp[i]['kp'], nkp[i - 1]['kp'],
                            nkp[i]['ts'] - nkp[i - 1]['ts']
                        )
                        for i in range(1, len(nkp))
                    ]
                    if nspeeds:
                        nearby_max  = max(nspeeds)
                        nearby_fast = sum(1 for s in nspeeds if s > self.VIOLENCE_SPEED_LOW)
                        if nearby_fast >= 3 and nearby_max > self.VIOLENCE_SPEED_LOW:
                            violence_type = 'mutual_altercation'
                            confidence    = max(confidence, 0.8)
                            involved_ids.append(nid)
                        elif nearby_fast < 2:
                            violence_type = 'assault'
                            confidence    = max(confidence, 0.7)
                            involved_ids.append(nid)

            speed_conf  = self._calculate_confidence(
                max_arm_speed, self.VIOLENCE_SPEED_LOW, self.VIOLENCE_SPEED_HIGH)
            consistency = fast_arm_frames / len(arm_speeds)
            confidence  = max(confidence, speed_conf * consistency)

            if confidence > 0.5:
                self._mark_behavior_alerted(pid, 'violence')
                for vid in involved_ids:
                    self.tracks[vid]['violent'] = True

                if FACE_BLUR and camera_id is not None:
                    try:
                        register_violence_event(camera_id, involved_ids)
                    except Exception as e:
                        print(f"[BehaviorDetector] Failed to register violence event: {e}")

                return {
                    'type':         'violence',
                    'subtype':      violence_type,
                    'person_id':    pid,
                    'involved_ids': involved_ids,
                    'position':     [int(curr_pos[0]), int(curr_pos[1])],
                    'confidence':   round(confidence, 2),
                    'max_speed':    round(max_arm_speed, 1),
                    'zone':         t['zone'].name if t['zone'] else 'unknown'
                }

        return None

    # ------------------------------------------------------------------
    # Detection: crowd formation
    # ------------------------------------------------------------------

    def detect_crowd(self, bboxes, ts):
        if not bboxes:
            return []

        MIN_CLUSTER = 3
        LINK_METERS = 2.5
        COOLDOWN    = 60.0

        centroids = [
            ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2, b)
            for b in bboxes
        ]
        n = len(centroids)
        if n < MIN_CLUSTER:
            return []

        neighbours = [set() for _ in range(n)]
        for i in range(n):
            xi, yi, bi = centroids[i]
            px_thresh = self.perspective.meters_to_pixels(LINK_METERS, yi)
            px_thresh = max(px_thresh, 40)
            for j in range(i + 1, n):
                xj, yj, _ = centroids[j]
                if math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) <= px_thresh:
                    neighbours[i].add(j)
                    neighbours[j].add(i)

        visited  = [False] * n
        clusters = []
        for start in range(n):
            if visited[start]:
                continue
            queue   = [start]
            members = []
            while queue:
                idx = queue.pop()
                if visited[idx]:
                    continue
                visited[idx] = True
                members.append(idx)
                queue.extend(neighbours[idx] - {i for i in range(n) if visited[i]})
            if len(members) >= MIN_CLUSTER:
                clusters.append(members)

        alerts = []
        for members in clusters:
            count = len(members)
            pts   = [(centroids[i][0], centroids[i][1]) for i in members]
            cx    = int(sum(p[0] for p in pts) / count)
            cy    = int(sum(p[1] for p in pts) / count)

            zone_id = f"crowd_{cx // 200}_{cy // 200}"
            if not self._can_alert_zone(zone_id, ts, cooldown=COOLDOWN):
                continue

            cxs     = [p[0] for p in pts]
            cys     = [p[1] for p in pts]
            hull_w  = max(cxs) - min(cxs) + 1
            hull_h  = max(cys) - min(cys) + 1
            density = round(count / (hull_w * hull_h) * 10_000, 2)

            confidence = min(
                0.5 + 0.5 * (count - MIN_CLUSTER) / max(2 * MIN_CLUSTER, 1),
                1.0
            )

            if confidence > 0.3:
                self.last_alerts[zone_id] = ts
                avg_area = int(
                    sum(centroids[i][2][2] * centroids[i][2][3] for i in members) / count
                )

                xs  = [centroids[i][2][0]                      for i in members]
                ys  = [centroids[i][2][1]                      for i in members]
                x2s = [centroids[i][2][0] + centroids[i][2][2] for i in members]
                y2s = [centroids[i][2][1] + centroids[i][2][3] for i in members]

                alerts.append({
                    'type':          'crowd_formation',
                    'count':         count,
                    'position':      [cx, cy],
                    'confidence':    round(confidence, 2),
                    'density':       density,
                    'avg_bbox_area': avg_area,
                    'hull_px':       [int(min(xs)),  int(min(ys)),
                                      int(max(x2s)), int(max(y2s))],
                })

        return alerts

    # ------------------------------------------------------------------
    # Main frame analysis
    # FIX: detection order changed — velocity-based checks run first and
    # mark 'running' alerted before gait runs, so gait is a true fallback.
    # ------------------------------------------------------------------

    def analyze_frame(self, persons, frame, ts, pose_results=None, camera_id=None):
        self.frame_count += 1
        self.ts_now = ts          # FIX: expose current timestamp to _can_alert_behavior
        alerts     = []
        all_bboxes = []
        all_pos    = {}

        if self.baseline_frames < 100:
            self.baseline_frames += 1

        for p in persons:
            pid, bbox = p['id'], p['bbox']
            all_bboxes.append(bbox)
            kp = self._extract_keypoints(pose_results, bbox)
            self.update_person(pid, bbox, ts, kp)
            all_pos[pid] = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

        for p in persons:
            pid, bbox = p['id'], p['bbox']

            # Fence-entry running (velocity already captured at entry time)
            if alert := self.detect_fence_entry_running(pid, ts):
                alerts.append(alert)

            # Loitering (checked every 30 frames to save CPU)
            if self.frame_count % 30 == 0:
                if alert := self.detect_loitering(pid, ts):
                    alerts.append(alert)

            # FIX: velocity-based detectors run BEFORE gait so that
            # _can_alert_behavior('running') is already consumed when
            # detect_running_by_gait is evaluated, preventing double-firing.
            if alert := self.detect_sprinting(pid, ts):
                alerts.append(alert)

            if alert := self.detect_running(pid, ts):
                alerts.append(alert)

            # Gait is the fallback — only fires if neither sprinting nor
            # speed-based running already marked the PID as alerted.
            if alert := self.detect_running_by_gait(pid, ts):
                alerts.append(alert)

            if alert := self.detect_violence(pid, ts, all_pos, camera_id=camera_id):
                alerts.append(alert)

        if self.frame_count % 20 == 0:
            alerts.extend(self.detect_crowd(all_bboxes, ts))

        return alerts

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def cleanup_old_tracks(self, active_ids, max_age=30.0):
        now       = time.time()
        to_remove = [
            pid for pid, t in self.tracks.items()
            if pid not in active_ids
            and t['last_seen'] is not None
            and (now - t['last_seen']) > max_age
        ]
        for pid in to_remove:
            del self.tracks[pid]

    def reset(self):
        self.tracks.clear()
        self.last_alerts.clear()
        self.velocity_history.clear()
        self.crowd_history.clear()
        self.baseline_frames = 0
        self.ts_now          = 0.0

        if FACE_BLUR:
            try:
                reset_violence_state()
            except Exception as e:
                print(f"[BehaviorDetector] Failed to reset violence state: {e}")

    def get_stats(self):
        active  = len(self.tracks)
        violent = sum(1 for t in self.tracks.values() if t['violent'])

        avg_velocity = np.mean(list(self.velocity_history)) if self.velocity_history else 0
        avg_crowd    = np.mean(list(self.crowd_history))    if self.crowd_history    else 0

        stats = {
            'active_tracks':   active,
            'violent_persons': violent,
            'total_alerts':    len(self.last_alerts),
            'avg_velocity':    round(avg_velocity, 2),
            'avg_crowd_size':  round(avg_crowd, 1),
            'calibration': {
                'px_per_meter':       round(self.camera.pixels_per_meter(), 2),
                'walk_threshold':     round(self.WALK_SPEED, 2),
                'run_threshold':      round(self.RUN_SPEED, 2),
                'run_consistency':    RUN_CONSISTENCY_MIN,
                'run_confidence_min': RUN_CONFIDENCE_MIN,
                'gait_min_signals':   GAIT_MIN_SIGNALS,
            }
        }

        if self.perspective.enabled:
            stats['perspective'] = {
                'enabled':       True,
                'horizon_y':     self.perspective.horizon_y,
                'depth_scaling': self.perspective.perspective.depth_scaling_power
            }
        else:
            stats['perspective'] = {'enabled': False}

        return stats


BehaviorDetector = AdaptiveBehaviorDetector
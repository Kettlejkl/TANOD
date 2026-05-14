import cv2
import time
import math
import logging
import signal
import atexit
import base64
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort

from app.extensions import socketio
from app.api.video.yolo_model import (
    model as yolo_pose_model,
    blur_faces,
    feed_detections_to_blurrer,
    face_model,
    FACE_CONF_THRESHOLD,
    FACE_IOU_THRESHOLD,
    FACE_IMG_SIZE,
    device as yolo_device,
)
from app.api.video.geo_fence import MultiGeoFenceManager
from app.api.video.stabilizer import ResponsiveBoxFilter
from app.api.video.behavior_detector import BehaviorDetector
from app.api.video.reid import create_tracker
from .sync_manager import SynchronizedVideoManager
from .db_handler import AnalyticsDBHandler
from .alerts_sender import AlertsSender
from .metrics_logger import MetricsLogger
from .video_processor import (
    hash_id, safe_get_track_confidence, convert_to_json_serializable,
    draw_crowd_alert,
    draw_crowd_hull,
    get_behavior_color,
    BOX_COLOR, LOITERING_COLOR, RUNNING_COLOR,
    VIOLENCE_COLOR, CROWD_COLOR
)

import sys
import os

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

try:
    from app.api.analytics.database import analytics_db
    print("[StreamManager] ✅ Analytics database imported successfully")
except ImportError as e:
    print(f"[StreamManager] ❌ Failed to import analytics_db: {e}")
    analytics_db = None


FACE_BLUR_ENABLED          = True
FACE_BLUR_INTERVAL         = 2

YOLO_INFERENCE_INTERVAL    = 3
BEHAVIOR_ANALYSIS_INTERVAL = 4
REID_INTERVAL              = 5
DB_SAVE_INTERVAL           = 12
FRAME_EMIT_INTERVAL        = 2
ENCODING_QUALITY           = 60
FRAME_RESIZE_WIDTH         = 640
REID_CLEANUP_AGE           = 60.0
DEBUG_TIMING               = True
TIMING_LOG_INTERVAL        = 30
METRICS_LOG_INTERVAL       = 10
DETECTION_EMIT_INTERVAL    = 25

CROWD_HULL_DISPLAY_SECONDS = 10.0

RUN_ENTRY_PX_PER_S         = 80
RUN_ENTRY_WINDOW_SEC       = 2.5

# ── Ghost box tunables ────────────────────────────────────────────────────────
# How long (seconds) the orange ghost box lingers after a fast exit
GHOST_LINGER_SEC           = 2.5
# Delay (seconds) before the ghost box becomes visible after the person exits.
# During this window the ghost exists in memory but is not drawn yet.
# Set to 0.0 to show immediately on exit.
GHOST_DELAY_SEC            = 0.5
# BGR colour for the ghost box border and label background (orange)
GHOST_COLOR_BGR            = (0, 140, 255)
# Label background colour (darker orange for contrast)
GHOST_LABEL_BGR            = (0, 100, 200)
# ─────────────────────────────────────────────────────────────────────────────

# Priority order for behavior colors (higher = drawn last = on top)
_BEHAVIOR_PRIORITY = {'loitering': 1, 'running': 2, 'sprinting': 3, 'violence': 4}


def encode_frame(frame, quality=60):
    h, w = frame.shape[:2]
    if w > FRAME_RESIZE_WIDTH:
        scale = FRAME_RESIZE_WIDTH / w
        frame = cv2.resize(frame, (FRAME_RESIZE_WIDTH, int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY,     quality,
        cv2.IMWRITE_JPEG_OPTIMIZE,    0,
        cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
    ]
    _, buf = cv2.imencode('.jpg', frame, encode_params)
    return base64.b64encode(buf).decode('utf-8')


def draw_box_clean(frame, ltrb, uid_short, color, _used_rects=[]):
    """
    Draw a person bounding box with a UID label.

    _used_rects is a mutable default — it accumulates label rectangles placed
    during this frame so labels can nudge downward to avoid collisions.
    Call  draw_box_clean.__defaults__[0].clear()  before each frame's draw loop
    to reset the per-frame state.
    """
    l, t, r, b = ltrb
    font      = cv2.FONT_HERSHEY_DUPLEX
    scale     = 0.38
    thickness = 1
    label     = f"UID {uid_short}"

    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad   = 4
    lbl_h = th + pad * 2
    lbl_w = tw + pad * 2

    lbl_x1 = max(l, 0)
    lbl_y2 = max(t, lbl_h)
    lbl_y1 = lbl_y2 - lbl_h

    for _ in range(8):
        candidate = (lbl_x1, lbl_y1, lbl_x1 + lbl_w, lbl_y2)
        collision = any(
            candidate[0] < ox2 and candidate[2] > ox1 and
            candidate[1] < oy2 and candidate[3] > oy1
            for (ox1, oy1, ox2, oy2) in _used_rects
        )
        if not collision:
            break
        lbl_y1 += lbl_h + 2
        lbl_y2 += lbl_h + 2

    _used_rects.append((lbl_x1, lbl_y1, lbl_x1 + lbl_w, lbl_y2))

    overlay = frame.copy()
    cv2.rectangle(overlay, (lbl_x1, lbl_y1), (lbl_x1 + lbl_w, lbl_y2),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    tx = lbl_x1 + pad
    ty = lbl_y1 + pad + th
    cv2.putText(frame, label, (tx + 1, ty + 1), font, scale, (0, 0, 0),
                thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (tx,     ty),     font, scale, color,
                thickness,     cv2.LINE_AA)

    cv2.rectangle(frame, (l, t), (r, b), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (l, t), (r, b), color,      1, cv2.LINE_AA)


class VideoStreamManager:

    def __init__(self):
        self.cameras                    = {}
        self.active_ids                 = {}
        self.sync_manager               = SynchronizedVideoManager()
        self.global_permanent_behaviors = {}
        self.active_crowd_hulls         = {}

        # ── Ghost boxes: camera_id → {track_id: ghost_dict} ──────────────────
        # Populated when a person exits a fence at running speed.
        # Each ghost_dict: {ltrb, pid, created_at, expires_at, speed_px_s}
        self._ghost_boxes               = {}
        # ─────────────────────────────────────────────────────────────────────

        if analytics_db is not None:
            self.db = AnalyticsDBHandler(analytics_db)
            print("[StreamManager] ✅ Analytics database handler initialized")
        else:
            self.db = None
            print("[StreamManager] ⚠️ Running without database")

        reid_config = {
            'feature_extractor': {
                'model_name': 'osnet_x1_0',
                'use_gpu':    True,
                'log_level':  logging.INFO
            },
            'appearance_analyzer': {'log_level': logging.INFO},
            'motion_tracker': {
                'kalman_enabled': True,
                'max_history':    40,
                'max_trajectory': 100,
                'log_level':      logging.INFO
            },
            'feature_matcher': {
                'similarity_threshold':        0.42,
                'cross_camera_threshold':      0.35,
                'spatial_proximity_threshold': 450,
                'spatial_proximity_bonus':     0.40,
                'spatial_time_window':         45.0,
                'color_weight':                0.35,
                'clothing_weight':             0.35,
                'velocity_weight':             0.15,
                'iou_weight':                  0.15,
                'continuity_bonus':            0.40,
                'min_feature_separation':      0.06,
                'log_level':                   logging.INFO
            },
            'person_database': {
                'max_features_per_person': 50,
                'max_tracked_persons':     200,
                'max_age':                 900.0,
                'log_level':               logging.INFO
            },
            'tracking_manager': {
                'min_box_area': 600,
                'log_level':    logging.INFO
            }
        }

        self.persistent_tracker = create_tracker(reid_config)
        print("[StreamManager] ✅ ReID tracking initialized")

        self.trackers             = {}
        self.stabilizers          = {}
        self.geo_fence_managers   = {}
        self.behavior_detectors   = {}
        self.fps_counters         = {}
        self._fence_entry_tracker = {}

        self._last_reid_results  = {}
        self._last_track_mapping = {}
        self._timing_accum       = {}
        self._timing_count       = {}

        self.frame_skip_ratio = 0.0
        self.alerts_sender    = AlertsSender()
        self.metrics_logger   = MetricsLogger()

        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("\n" + "=" * 60)
        print("🎯 STREAM MANAGER")
        print("=" * 60)
        print(f"  😶 Face Blur  : every {FACE_BLUR_INTERVAL} frames")
        print(f"  🔍 YOLO       : every {YOLO_INFERENCE_INTERVAL} frames")
        print(f"  🧠 Behavior   : every {BEHAVIOR_ANALYSIS_INTERVAL} frames")
        print(f"  🔁 ReID       : every {REID_INTERVAL} frames")
        print(f"  📤 Emit       : every {FRAME_EMIT_INTERVAL} frames")
        print(f"  🏃 Run Entry  : >{RUN_ENTRY_PX_PER_S} px/s within {RUN_ENTRY_WINDOW_SEC}s")
        print(f"  👻 Ghost box  : {GHOST_DELAY_SEC}s delay → {GHOST_LINGER_SEC}s linger on fast exit")
        print("=" * 60 + "\n")

    def _signal_handler(self, signum, frame):
        print(f"\n[StreamManager] ⚠️ Signal {signum}, shutting down...")
        self.shutdown(); sys.exit(0)

    def shutdown(self):
        print("\n[StreamManager] 🛑 Initiating shutdown...")
        for cid in list(self.cameras):
            if self.cameras[cid]['active']:
                self.stop_stream(cid)
        try:   self.alerts_sender.stop()
        except Exception as e: print(f"  ⚠️ {e}")
        try:
            if hasattr(self, 'metrics_logger'):
                self.metrics_logger.finalize()
        except Exception as e: print(f"  ❌ Metrics: {e}")
        print("[StreamManager] ✅ Shutdown complete\n")

    def _timing_reset(self, cid):
        self._timing_accum[cid] = {
            'yolo':0., 'face_det':0., 'deepsort':0.,
            'reid':0., 'behavior':0., 'draw':0.,
            'blur':0., 'encode_emit':0., 'total':0.,
        }
        self._timing_count[cid] = 0

    def _timing_add(self, cid, stage, ms):
        self._timing_accum[cid][stage] += ms

    def _timing_report(self, cid, frame_count):
        self._timing_count[cid] += 1
        if self._timing_count[cid] < TIMING_LOG_INTERVAL:
            return
        n = self._timing_count[cid]; acc = self._timing_accum[cid]
        print(f"\n{'─'*70}")
        print(f"[TIMING] {cid} — avg over {n} frames  (frame #{frame_count})")
        for stage, total in acc.items():
            avg_ms = total / n
            print(f"  {stage:<14} {avg_ms:6.1f} ms  {'█'*int(avg_ms/5)}")
        print(f"{'─'*70}\n")
        self._timing_reset(cid)

    _DEEPSORT_PARAMS = dict(
        max_age=60,
        n_init=1,
        max_iou_distance=0.95,
        max_cosine_distance=0.55,
        nn_budget=600,
    )

    def add_camera(self, camera_id, source, start_offset_sec=0.0, loop=False):
        self.cameras[camera_id] = {
            'source': source, 'cap': None, 'active': False,
            'start_offset_sec': start_offset_sec,
            'loop': loop,
        }
        self.active_ids[camera_id]             = set()
        self.geo_fence_managers[camera_id]     = MultiGeoFenceManager()
        self.trackers[camera_id]               = DeepSort(**self._DEEPSORT_PARAMS)
        self.stabilizers[camera_id]            = ResponsiveBoxFilter()
        self.behavior_detectors[camera_id]     = BehaviorDetector()
        self.fps_counters[camera_id]           = {
            'count': 0, 'start_time': time.time(), 'fps': 0}
        self._last_reid_results[camera_id]     = []
        self._last_track_mapping[camera_id]    = {}
        self._fence_entry_tracker[camera_id]   = {}
        self._ghost_boxes[camera_id]           = {}
        self._timing_reset(camera_id)
        print(f"[StreamManager] Camera {camera_id} added")

    def add_geo_fence(self, camera_id, name, points):
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found"); return None
        fid = self.geo_fence_managers[camera_id].add_fence(name, points)
        if fid:
            print(f"[StreamManager] Added geo-fence '{name}' ({fid}) to {camera_id}")
        return fid

    def remove_geo_fence(self, camera_id, fence_id):
        if camera_id not in self.geo_fence_managers:
            return False
        self.geo_fence_managers[camera_id].remove_fence(fence_id); return True

    def update_geo_fence(self, camera_id, fence_id, points=None, name=None, enabled=None):
        if camera_id not in self.geo_fence_managers:
            return False
        return self.geo_fence_managers[camera_id].update_fence(
            fence_id, points, name, enabled)

    def toggle_geo_fence(self, camera_id, fence_id):
        if camera_id not in self.geo_fence_managers:
            return None
        return self.geo_fence_managers[camera_id].toggle_fence(fence_id)

    def get_geo_fences(self, camera_id):
        if camera_id in self.geo_fence_managers:
            return self.geo_fence_managers[camera_id].get_all_fences()
        return []

    def load_geo_fences_from_config(self, camera_id, fences_config):
        if camera_id not in self.geo_fence_managers:
            return False
        self.geo_fence_managers[camera_id].load_from_config(fences_config); return True

    def _update_fps(self, camera_id):
        c = self.fps_counters[camera_id]; c['count'] += 1
        if c['count'] % 30 == 0:
            c['fps'] = 30 / (time.time() - c['start_time'])
            c['start_time'] = time.time()

    def start_stream(self, camera_id):
        if camera_id not in self.cameras: return False
        cam = self.cameras[camera_id]
        if cam['active']: return True
        cam['cap'] = cv2.VideoCapture(cam['source'])
        cam['active'] = True
        socketio.start_background_task(self._stream_frames, camera_id)
        return True

    def stop_stream(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id]['active'] = False
            if self.cameras[camera_id]['cap']:
                self.cameras[camera_id]['cap'].release()

    def soft_reset_camera(self, camera_id):
        if camera_id not in self.trackers: return False
        self.trackers[camera_id] = DeepSort(**self._DEEPSORT_PARAMS)
        self.stabilizers[camera_id].cleanup(set())
        self.behavior_detectors[camera_id].cleanup_old_tracks(set(), max_age=0)
        self.active_ids[camera_id]           = set()
        self._last_reid_results[camera_id]   = []
        self._last_track_mapping[camera_id]  = {}
        self._fence_entry_tracker[camera_id] = {}
        self._ghost_boxes[camera_id]         = {}
        self.active_crowd_hulls.pop(camera_id, None)
        print(f"[StreamManager] Soft reset {camera_id}")
        return True

    def _check_fence_entry_running(
        self, camera_id, track_id, centroid, current_time,
        fence_id, fence_name, track_mapping, frame_count
    ):
        entry_tracker   = self._fence_entry_tracker[camera_id]
        behavior_det    = self.behavior_detectors[camera_id]
        persistent_id   = track_mapping.get(track_id)

        if track_id not in entry_tracker:
            entry_tracker[track_id] = {
                'entered_at': current_time,
                'entry_pos':  centroid,
                'recorded':   False,
            }
            return

        info = entry_tracker[track_id]
        if info['recorded']:
            return

        dt = current_time - info['entered_at']

        if dt < 0.1 or dt > RUN_ENTRY_WINDOW_SEC:
            return

        dx = centroid[0] - info['entry_pos'][0]
        dy = centroid[1] - info['entry_pos'][1]
        pixel_dist     = math.sqrt(dx * dx + dy * dy)
        speed_px_per_s = pixel_dist / dt

        info['recorded'] = True
        if persistent_id is not None:
            behavior_det.record_fence_entry(
                pid        = persistent_id,
                speed_px_s = speed_px_per_s,
                centroid   = centroid,
                ts         = current_time,
                fence_name = fence_name,
                fence_id   = fence_id,
                bbox_y     = float(centroid[1]),
            )

    def _get_velocity_for_track(self, camera_id, track_id):
        track_mapping = self._last_track_mapping.get(camera_id, {})
        persistent_id = track_mapping.get(track_id)
        if persistent_id is None:
            return (0.0, 0.0)
        bd = self.behavior_detectors.get(camera_id)
        if bd is None:
            return (0.0, 0.0)
        t = bd.tracks.get(persistent_id)
        if t is None or len(t['pos']) < 2 or len(t['ts']) < 2:
            return (0.0, 0.0)
        pos_list = list(t['pos'])
        ts_list  = list(t['ts'])
        dx = pos_list[-1][0] - pos_list[-2][0]
        dy = pos_list[-1][1] - pos_list[-2][1]
        dt = ts_list[-1] - ts_list[-2]
        if dt <= 0:
            return (0.0, 0.0)
        return (dx / dt, dy / dt)

    def _extrapolate_detections(self, camera_id, detections, frames_elapsed):
        if frames_elapsed <= 1 or not detections:
            return detections
        shift   = min(frames_elapsed, 2)
        result  = []
        for (bbox, conf, cls) in detections:
            x, y, w, h = bbox
            cx = x + w / 2
            cy = y + h / 2
            best_vel  = (0.0, 0.0)
            best_dist = 80.0
            bd = self.behavior_detectors.get(camera_id)
            if bd is not None:
                for pid, tr in bd.tracks.items():
                    if len(tr['pos']) < 2 or len(tr['ts']) < 2:
                        continue
                    pos_list = list(tr['pos'])
                    ts_list  = list(tr['ts'])
                    tx, ty   = pos_list[-1]
                    dist = math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        dt = ts_list[-1] - ts_list[-2]
                        if dt > 0:
                            best_vel = (
                                (pos_list[-1][0] - pos_list[-2][0]) / dt,
                                (pos_list[-1][1] - pos_list[-2][1]) / dt,
                            )
            vx, vy  = best_vel
            x_new   = x + vx * shift
            y_new   = y + vy * shift
            result.append(([x_new, y_new, w, h], conf, cls))
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # GHOST BOX RENDERER
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_ghost_boxes(self, display_frame, camera_id, current_time):
        """
        Render all active ghost boxes for this camera onto display_frame.

        Ghost boxes appear when a person exits a geo-fence at running speed.
        They are hidden for GHOST_DELAY_SEC after exit (delay window), then
        fade from full opacity to transparent over GHOST_LINGER_SEC seconds,
        coloured orange with a "Running" label showing the exit speed.

        Timeline per ghost:
          created_at --[delay]--> visible_at --[linger]--> expires_at
                       invisible              fading 1->0

        Returns the list of track_ids whose ghosts have expired so the caller
        can remove them.
        """
        cam_ghosts   = self._ghost_boxes.get(camera_id, {})
        expired_tids = []

        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.38

        for ghost_tid, ghost in cam_ghosts.items():
            # Hard expiry — remove from dict next iteration
            if current_time > ghost['expires_at']:
                expired_tids.append(ghost_tid)
                continue

            # Still inside the delay window — ghost exists but is invisible
            if current_time < ghost['visible_at']:
                continue

            # Inside the linger window — compute fade alpha (1.0 -> 0.0)
            linger_elapsed = current_time - ghost['visible_at']
            linger_total   = ghost['expires_at'] - ghost['visible_at']
            alpha = max(0.0, 1.0 - (linger_elapsed / max(linger_total, 1e-6)))

            lx, ty_, rx, by = ghost['ltrb']
            speed_px_s      = ghost['speed_px_s']
            label           = f"Running  {speed_px_s:.0f} px/s"

            overlay = display_frame.copy()

            # Draw dashed orange bounding box (simulate dashes with segments)
            dash_len = 12
            gap_len  = 6
            pts = [
                ((lx, ty_), (rx, ty_)),   # top
                ((rx, ty_), (rx, by)),     # right
                ((rx, by),  (lx, by)),     # bottom
                ((lx, by),  (lx, ty_)),    # left
            ]
            for (x1, y1), (x2, y2) in pts:
                seg_len  = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                n_steps  = max(1, int(seg_len / (dash_len + gap_len)))
                for i in range(n_steps):
                    t_start = i * (dash_len + gap_len) / seg_len
                    t_end   = min((i * (dash_len + gap_len) + dash_len) / seg_len, 1.0)
                    sx = int(x1 + (x2 - x1) * t_start)
                    sy = int(y1 + (y2 - y1) * t_start)
                    ex = int(x1 + (x2 - x1) * t_end)
                    ey = int(y1 + (y2 - y1) * t_end)
                    cv2.line(overlay, (sx, sy), (ex, ey),
                             GHOST_COLOR_BGR, 2, cv2.LINE_AA)

            # Label background + text (above the box)
            (tw, th_), _ = cv2.getTextSize(label, font, scale, 1)
            pad    = 4
            lx1    = max(lx, 0)
            ly2    = max(ty_ - 2, th_ + pad * 2)
            ly1    = ly2 - (th_ + pad * 2)
            lx2    = lx1 + tw + pad * 2

            cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), GHOST_LABEL_BGR, -1)
            cv2.putText(overlay, label,
                        (lx1 + pad, ly2 - pad),
                        font, scale, (255, 255, 255), 1, cv2.LINE_AA)

            # Blend ghost onto the real frame using current alpha
            cv2.addWeighted(overlay, alpha, display_frame, 1.0 - alpha,
                            0, display_frame)

        return expired_tids

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN STREAM LOOP
    # ─────────────────────────────────────────────────────────────────────────

    def _stream_frames(self, camera_id):
        camera            = self.cameras[camera_id]
        cap               = camera['cap']
        frame_count       = 0
        geo_fence_manager = self.geo_fence_managers[camera_id]
        tracker           = self.trackers[camera_id]
        stabilizer        = self.stabilizers[camera_id]
        behavior_detector = self.behavior_detectors[camera_id]
        entry_tracker     = self._fence_entry_tracker[camera_id]

        ret, _ = cap.read()
        if ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        last_yolo_detections = []
        last_pose_results    = None
        last_clean_frame     = None

        total_frames       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps                = cap.get(cv2.CAP_PROP_FPS) or 30.0
        is_video_file      = total_frames > 0
        video_duration_sec = total_frames / fps if fps > 0 else 0
        frame_times        = []

        print(f"\n[StreamManager] 🎥 Starting {camera_id}  "
              f"({total_frames:,} frames @ {fps:.1f} fps — "
              f"{video_duration_sec/60:.1f} min)\n")

        try:
            import torch
            if torch.cuda.is_available():
                print(f"[StreamManager] 🖥️  CUDA: {torch.cuda.get_device_name(0)}")
            else:
                print("[StreamManager] ⚠️  No CUDA — running on CPU")
        except Exception:
            pass

        while camera['active'] and cap.isOpened():
            try:
                t_loop_start = time.perf_counter()

                ret, frame = cap.read()

                if not ret:
                    if camera['loop']:
                        print(f"[StreamManager] 🔁 Looping {camera_id}")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            print(f"[StreamManager] Loop failed, stopping {camera_id}")
                            break
                    else:
                        print(f"[StreamManager] "
                              f"{'Video ended' if is_video_file else 'Stream lost'}: "
                              f"{camera_id}")
                        break

                frame_count  += 1
                current_time  = time.time()
                self._update_fps(camera_id)

                if frame.shape[1] > FRAME_RESIZE_WIDTH:
                    aspect = frame.shape[0] / frame.shape[1]
                    frame  = cv2.resize(
                        frame,
                        (FRAME_RESIZE_WIDTH, int(FRAME_RESIZE_WIDTH * aspect)),
                        interpolation=cv2.INTER_AREA)

                clean_frame   = frame.copy()
                display_frame = frame.copy()

                # ── YOLO inference ────────────────────────────────────────────
                t0 = time.perf_counter()
                if frame_count % YOLO_INFERENCE_INTERVAL == 0:
                    results      = yolo_pose_model(
                        clean_frame, conf=0.20, iou=0.45,
                        imgsz=640, verbose=False, max_det=50)
                    detections   = []
                    pose_results = results

                    if len(results[0].boxes) > 0:
                        boxes   = results[0].boxes.xyxy.cpu().numpy()
                        confs   = results[0].boxes.conf.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy()
                        indices = cv2.dnn.NMSBoxes(
                            [b.tolist() for b in boxes],
                            confs.tolist(), 0.20, 0.70)
                        if len(indices) > 0:
                            indices = indices.flatten()
                            boxes   = boxes[indices]
                            confs   = confs[indices]
                            classes = classes[indices]
                        for box, conf, cls in zip(boxes, confs, classes):
                            if int(cls) == 0 and conf > 0.12:
                                lx, ty_, rx, by = map(int, box)
                                if (rx - lx) * (by - ty_) >= 500:
                                    detections.append(
                                        ([lx, ty_, rx-lx, by-ty_],
                                         float(conf), 'person'))

                    last_yolo_detections = detections
                    last_pose_results    = pose_results
                    last_clean_frame     = clean_frame.copy()
                else:
                    frames_since_yolo = frame_count % YOLO_INFERENCE_INTERVAL
                    detections   = self._extrapolate_detections(
                        camera_id, last_yolo_detections, frames_since_yolo)
                    pose_results = last_pose_results

                t1 = time.perf_counter()
                if DEBUG_TIMING:
                    self._timing_add(camera_id, 'yolo', (t1 - t0) * 1000)

                # ── Face detection / blur feed ────────────────────────────────
                if FACE_BLUR_ENABLED and frame_count % FACE_BLUR_INTERVAL == 0:
                    t0 = time.perf_counter()
                    face_results = None
                    if face_model is not None:
                        try:
                            face_results = face_model(
                                clean_frame,
                                conf=FACE_CONF_THRESHOLD,
                                iou=FACE_IOU_THRESHOLD,
                                verbose=False,
                                imgsz=FACE_IMG_SIZE,
                                device=yolo_device,
                                half=(yolo_device == 'cuda'),
                            )
                        except Exception as fe:
                            print(f"[StreamManager] face_model error: {fe}")
                    feed_detections_to_blurrer(
                        camera_id,
                        face_results,
                        pose_results=last_pose_results,
                    )
                    t1 = time.perf_counter()
                    if DEBUG_TIMING:
                        self._timing_add(camera_id, 'face_det', (t1 - t0) * 1000)

                # ── DeepSORT tracking ─────────────────────────────────────────
                t0     = time.perf_counter()
                tracks = tracker.update_tracks(detections, frame=clean_frame)
                tracks = [tk for tk in tracks
                          if tk.is_confirmed() and tk.time_since_update < 12]
                t1     = time.perf_counter()
                if DEBUG_TIMING:
                    self._timing_add(camera_id, 'deepsort', (t1 - t0) * 1000)

                current_frame_ids          = set()
                current_track_ids          = set()
                inside_count               = 0
                person_tracks_for_analysis = []
                reid_detections            = []

                inside_track_ids = set()

                for track in tracks:
                    track_id = track.track_id
                    lx, ty_, rx, by = map(int, track.to_ltrb())
                    is_inside, fence_id, fence_name = \
                        geo_fence_manager.is_person_inside_any((lx, ty_, rx, by))

                    if not is_inside:
                        if track_id in entry_tracker:
                            # ── Ghost box on fast exit ────────────────────────
                            vel        = self._get_velocity_for_track(camera_id, track_id)
                            speed_px_s = math.sqrt(vel[0] ** 2 + vel[1] ** 2)

                            if speed_px_s >= RUN_ENTRY_PX_PER_S:
                                persistent_id = self._last_track_mapping[camera_id].get(
                                    track_id)
                                self._ghost_boxes[camera_id][track_id] = {
                                    'ltrb':        (lx, ty_, rx, by),
                                    'pid':         persistent_id,
                                    'created_at':  current_time,
                                    # Ghost becomes visible after GHOST_DELAY_SEC
                                    'visible_at':  current_time + GHOST_DELAY_SEC,
                                    # Ghost disappears after delay + linger
                                    'expires_at':  current_time + GHOST_DELAY_SEC + GHOST_LINGER_SEC,
                                    'speed_px_s':  speed_px_s,
                                }
                                print(f"[GHOST] Track {track_id} exited at "
                                      f"{speed_px_s:.0f} px/s — ghost in "
                                      f"{GHOST_DELAY_SEC}s, linger {GHOST_LINGER_SEC}s")

                            del entry_tracker[track_id]
                            pid_exited = self._last_track_mapping[camera_id].get(track_id)
                            if pid_exited:
                                behavior_detector.clear_fence_entry(pid_exited)
                            # ─────────────────────────────────────────────────
                        continue

                    inside_count += 1
                    inside_track_ids.add(track_id)
                    current_track_ids.add(track_id)

                    centroid = ((lx + rx) // 2, (ty_ + by) // 2)

                    track_mapping_snap = self._last_track_mapping[camera_id]
                    self._check_fence_entry_running(
                        camera_id, track_id, centroid, current_time,
                        fence_id, fence_name, track_mapping_snap, frame_count,
                    )

                    raw_bbox = [lx, ty_, rx-lx, by-ty_]
                    bbox = raw_bbox
                    if hasattr(stabilizer, 'update'):
                        try:
                            stab = stabilizer.update(track_id, raw_bbox)
                            if stab:
                                bbox = list(stab) if isinstance(stab, tuple) else stab
                        except Exception:
                            pass
                    reid_detections.append({
                        'bbox': bbox, 'track_id': track_id,
                        'confidence': safe_get_track_confidence(track),
                    })

                # ── ReID ──────────────────────────────────────────────────────
                t0       = time.perf_counter()
                run_reid = (frame_count % REID_INTERVAL == 0
                            and len(reid_detections) > 0)
                if run_reid:
                    rf = last_clean_frame if last_clean_frame is not None else clean_frame
                    reid_results = self.persistent_tracker.process_detections(
                        camera_id=camera_id, detections=reid_detections,
                        frame=rf, timestamp=current_time)
                    self._last_reid_results[camera_id] = reid_results
                    self._last_track_mapping[camera_id].update(
                        {r['track_id']: r['persistent_id'] for r in reid_results})
                else:
                    reid_results = self._last_reid_results[camera_id]
                track_mapping = self._last_track_mapping[camera_id]
                t1 = time.perf_counter()
                if DEBUG_TIMING:
                    self._timing_add(camera_id, 'reid', (t1 - t0) * 1000)

                for result in reid_results:
                    track_id      = result['track_id']
                    persistent_id = result['persistent_id']
                    bbox          = result['bbox']
                    method        = result['method']
                    confidence    = result['confidence']
                    if (track_id in track_mapping
                            and track_mapping[track_id] != persistent_id):
                        print(f"⚠️ Track {track_id} → multiple IDs, skipping")
                        continue
                    current_frame_ids.add(persistent_id)
                    lx, ty_, w, h = bbox
                    is_inside, fence_id, fence_name = \
                        geo_fence_manager.is_person_inside_any(
                            (lx, ty_, lx+w, ty_+h))
                    if self.db and frame_count % DB_SAVE_INTERVAL == 0:
                        self.db.save_person_track(
                            camera_id=camera_id, persistent_id=persistent_id,
                            track_id=track_id, bbox=bbox,
                            metadata={
                                'confidence': confidence,
                                'in_geo_fence': is_inside,
                                'fence_id': fence_id,
                                'fence_name': fence_name,
                                'frame_id': frame_count,
                                'method': method,
                            })
                    person_tracks_for_analysis.append(
                        {'id': persistent_id, 'bbox': bbox})

                # ── Behavior analysis ─────────────────────────────────────────
                t0              = time.perf_counter()
                behavior_alerts = []
                if frame_count % BEHAVIOR_ANALYSIS_INTERVAL == 0:
                    behavior_alerts = behavior_detector.analyze_frame(
                        person_tracks_for_analysis, clean_frame, current_time,
                        pose_results=pose_results, camera_id=camera_id)
                t1 = time.perf_counter()
                if DEBUG_TIMING:
                    self._timing_add(camera_id, 'behavior', (t1 - t0) * 1000)

                for alert in behavior_alerts:
                    alert_type = alert['type']
                    alert      = convert_to_json_serializable(alert)
                    if 'person_id' in alert:
                        person_id = alert['person_id']
                        hashed    = hash_id(person_id)
                        behavior_map = {
                            'loitering': ('loitering', LOITERING_COLOR),
                            'running':   ('running',   RUNNING_COLOR),
                            'sprinting': ('running',   RUNNING_COLOR),
                            'violence':  ('violence',  VIOLENCE_COLOR),
                        }
                        if alert_type in behavior_map:
                            btype, bcolor = behavior_map[alert_type]
                            existing = self.global_permanent_behaviors.get(person_id)
                            ep  = _BEHAVIOR_PRIORITY.get(
                                existing['type'], 0) if existing else 0
                            np_ = _BEHAVIOR_PRIORITY.get(btype, 0)
                            if np_ >= ep:
                                self.global_permanent_behaviors[person_id] = {
                                    'type': btype, 'color': bcolor,
                                    'timestamp': current_time,
                                    'camera_id': camera_id, 'alert': alert,
                                }
                                print(f"[{alert_type.upper()}] UID {person_id} ({hashed})")
                        if self.db:
                            self.db.save_behavior_event(
                                camera_id=camera_id, persistent_id=person_id,
                                track_id=None, behavior_type=alert_type,
                                severity=alert.get('severity', 'medium'),
                                confidence=alert.get('confidence', 0.8),
                                description=f"{alert_type} detected",
                                metadata={k: v for k, v in alert.items()
                                          if k not in ['type','severity',
                                                       'description','position']},
                                position=alert.get('position'))
                    elif alert_type == 'crowd_formation':
                        print(f"🟡 [CROWD] {alert['count']} people")
                        if 'hull_px' in alert:
                            self.active_crowd_hulls[camera_id] = {
                                'hull_px':    alert['hull_px'],
                                'count':      alert['count'],
                                'confidence': alert.get('confidence', 0.8),
                                'expires_at': current_time + CROWD_HULL_DISPLAY_SECONDS,
                            }
                        if self.db:
                            self.db.save_behavior_event(
                                camera_id=camera_id, persistent_id=None,
                                track_id=None, behavior_type='crowd_formation',
                                severity='medium',
                                confidence=alert.get('confidence', 0.8),
                                description=f"Crowd of {alert['count']} detected",
                                metadata=alert, position=alert.get('position'))
                    self.alerts_sender.send({
                        'alert_type': 'behavior', 'type': alert_type,
                        'severity': alert.get('severity', 'medium'),
                        'location': camera_id,
                        'description': f"{alert_type} detected",
                        'metadata': {
                            'camera_id': camera_id, 'frame_id': frame_count,
                            'timestamp': datetime.now().isoformat(),
                            'confidence': alert.get('confidence', 0.8),
                            'person_id': alert.get('person_id'),
                            'position': alert.get('position'),
                            'details': {k: v for k, v in alert.items()
                                        if k not in ['type','severity',
                                                     'description','position']},
                        },
                    })

                # ── Drawing ───────────────────────────────────────────────────
                t0 = time.perf_counter()

                draw_list = []
                for track in tracks:
                    track_id      = track.track_id
                    persistent_id = track_mapping.get(track_id)
                    if persistent_id is None:
                        continue
                    lx, ty_, rx, by = map(int, track.to_ltrb())
                    is_inside, _, _ = geo_fence_manager.is_person_inside_any(
                        (lx, ty_, rx, by))
                    if not is_inside:
                        continue
                    color    = BOX_COLOR
                    priority = 0
                    if persistent_id in self.global_permanent_behaviors:
                        btype    = self.global_permanent_behaviors[persistent_id]['type']
                        color    = get_behavior_color(btype)
                        priority = _BEHAVIOR_PRIORITY.get(btype, 0)
                    draw_list.append((priority, track, persistent_id, color))

                draw_list.sort(key=lambda x: x[0])
                draw_box_clean.__defaults__[0].clear()

                for _, track, persistent_id, color in draw_list:
                    lx, ty_, rx, by = map(int, track.to_ltrb())
                    draw_box_clean(
                        display_frame,
                        (lx, ty_, rx, by),
                        uid_short=hash_id(persistent_id)[:6],
                        color=color,
                    )

                t1 = time.perf_counter()
                if DEBUG_TIMING:
                    self._timing_add(camera_id, 'draw', (t1 - t0) * 1000)

                # ── Ghost box overlay (fast-exit running indicators) ───────────
                expired_ghost_tids = self._draw_ghost_boxes(
                    display_frame, camera_id, current_time)
                for tid in expired_ghost_tids:
                    self._ghost_boxes[camera_id].pop(tid, None)

                # ── Crowd hull overlay ────────────────────────────────────────
                hull_info = self.active_crowd_hulls.get(camera_id)
                if hull_info:
                    if current_time < hull_info['expires_at']:
                        draw_crowd_hull(
                            display_frame,
                            hull_info['hull_px'],
                            hull_info['count'],
                            hull_info['confidence'],
                        )
                    else:
                        del self.active_crowd_hulls[camera_id]

                # ── Face blur ─────────────────────────────────────────────────
                t0 = time.perf_counter()
                if FACE_BLUR_ENABLED and frame_count % FACE_BLUR_INTERVAL == 0:
                    display_frame = blur_faces(display_frame, camera_id=camera_id)
                t1 = time.perf_counter()
                if DEBUG_TIMING:
                    self._timing_add(camera_id, 'blur', (t1 - t0) * 1000)

                # ── Housekeeping ──────────────────────────────────────────────
                if self.db and frame_count % 150 == 0:
                    self.db.deactivate_tracks(camera_id, list(current_frame_ids))

                all_active = set()
                for cid2 in self.cameras:
                    all_active.update(self.active_ids.get(cid2, set()))
                stale = [
                    pid for pid, beh in self.global_permanent_behaviors.items()
                    if pid not in all_active
                    and (current_time - beh.get('timestamp', current_time)) > 30.0
                ]
                for pid in stale:
                    del self.global_permanent_behaviors[pid]

                active_tids = {tk.track_id for tk in tracks}
                for tid in [tid for tid in list(self._last_track_mapping[camera_id])
                             if tid not in active_tids]:
                    del self._last_track_mapping[camera_id][tid]

                behavior_detector.cleanup_old_tracks(
                    current_frame_ids, max_age=REID_CLEANUP_AGE)
                stabilizer.cleanup(current_track_ids)
                self.persistent_tracker.cleanup_old_tracks(max_age=REID_CLEANUP_AGE)
                self.active_ids[camera_id] = current_frame_ids

                stale_entry_ids = [
                    tid for tid in list(entry_tracker)
                    if tid not in active_tids
                ]
                for tid in stale_entry_ids:
                    del entry_tracker[tid]

                # ── Detection emit ────────────────────────────────────────────
                if tracks and frame_count % DETECTION_EMIT_INTERVAL == 0:
                    det_payload = {
                        'camera_id': camera_id, 'frame_id': frame_count,
                        'timestamp': datetime.now().isoformat(), 'detections': [],
                    }
                    for track in tracks:
                        tid = track.track_id
                        lx, ty_, rx, by = map(int, track.to_ltrb())
                        conf = safe_get_track_confidence(track, default=0.8)
                        det_payload['detections'].append({
                            'class': 'person', 'track_id': int(tid),
                            'centroid': [int((lx+rx)//2), int((ty_+by)//2)],
                            'bbox': [int(lx), int(ty_), int(rx-lx), int(by-ty_)],
                            'confidence': float(conf),
                        })
                    self.alerts_sender.send(det_payload)

                if self.db and frame_count % 450 == 0:
                    try:
                        analytics_db.update_hourly_stats(camera_id, datetime.utcnow())
                    except Exception as e:
                        print(f"[StreamManager] Hourly stats error: {e}")

                # ── Frame emit ────────────────────────────────────────────────
                t0 = time.perf_counter()
                if frame_count % FRAME_EMIT_INTERVAL == 0:
                    frame_b64 = encode_frame(display_frame, quality=ENCODING_QUALITY)
                    socketio.emit('video_frame', {
                        'camera_id':    camera_id,
                        'frame':        frame_b64,
                        'timestamp':    time.time(),
                        'count':        len(current_frame_ids),
                        'inside_count': inside_count,
                        'permanent_behaviors_count':
                            len(self.global_permanent_behaviors),
                        'tracker_stats':
                            self.persistent_tracker.get_statistics(),
                        'fps':      self.fps_counters[camera_id]['fps'],
                        'progress': (frame_count / total_frames * 100)
                                    if total_frames > 0 else 0,
                    }, room=f'camera_{camera_id}')
                t1 = time.perf_counter()
                if DEBUG_TIMING:
                    self._timing_add(camera_id, 'encode_emit', (t1 - t0) * 1000)

                t_loop_end    = time.perf_counter()
                frame_latency = t_loop_end - t_loop_start
                if DEBUG_TIMING:
                    self._timing_add(camera_id, 'total', frame_latency * 1000)
                    self._timing_report(camera_id, frame_count)

                frame_times.append(frame_latency)
                if len(frame_times) > 100: frame_times.pop(0)

                if frame_count % METRICS_LOG_INTERVAL == 0:
                    avg_conf = (sum(d[1] for d in detections) / len(detections)
                                if detections else 0)
                    self.metrics_logger.log_frame(
                        camera_id=camera_id,
                        fps=1.0/frame_latency if frame_latency > 0 else 0,
                        detection_count=len(detections),
                        tracking_count=len(current_frame_ids),
                        latency=frame_latency,
                        avg_confidence=avg_conf,
                        occupancy=inside_count,
                    )

                if frame_count % 150 == 0:
                    stats      = self.persistent_tracker.get_statistics()
                    progress   = (frame_count/total_frames*100
                                  if total_frames > 0 else 0)
                    recent     = frame_times[-30:] if frame_times else [1]
                    actual_fps = 1.0 / (sum(recent) / len(recent))
                    mc = {}
                    for r in reid_results:
                        mc[r['method']] = mc.get(r['method'], 0) + 1
                    ghost_count = len(self._ghost_boxes.get(camera_id, {}))
                    print(f"\n[StreamManager] {camera_id} — frame "
                          f"{frame_count:,}/{total_frames:,} ({progress:.1f}%)")
                    print(f"  ⚡ Actual FPS : {actual_fps:.1f}")
                    print(f"  👥 Persons   : {stats['total_persons']}")
                    print(f"  🔁 ReID      : {mc}")
                    print(f"  📊 Match rate: {stats.get('match_rate', 0):.2f}")
                    print(f"  👻 Ghosts    : {ghost_count}")

                socketio.sleep(0.001)

            except Exception as e:
                print(f"[ERROR] {camera_id} frame {frame_count}: {e}")
                import traceback; traceback.print_exc()
                socketio.sleep(0.1); continue

        print(f"\n[StreamManager] 🛑 {camera_id} ended (frame {frame_count:,})")
        if frame_times:
            rec = frame_times[-30:] if len(frame_times) >= 30 else frame_times
            print(f"  Final FPS: {1.0/(sum(rec)/len(rec)):.1f}")
        cap.release(); camera['active'] = False

    def get_tracking_statistics(self):
        if hasattr(self, 'persistent_tracker'):
            stats = self.persistent_tracker.get_statistics()
            stats['global_permanent_behaviors'] = len(self.global_permanent_behaviors)
            return stats
        return None

    def get_permanent_behaviors(self):
        return {
            pid: {
                'type':      beh['type'],
                'hashed_id': hash_id(pid),
                'camera_id': beh['camera_id'],
                'timestamp': beh['timestamp'],
            }
            for pid, beh in self.global_permanent_behaviors.items()
        }

    def clear_permanent_behavior(self, persistent_id):
        if persistent_id in self.global_permanent_behaviors:
            del self.global_permanent_behaviors[persistent_id]
            print(f"🗑️ Cleared behavior UID {persistent_id} ({hash_id(persistent_id)})")
            return True
        return False
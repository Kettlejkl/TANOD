import cv2
import base64
import time
import numpy as np
import requests
from datetime import datetime
from app.extensions import socketio
from deep_sort_realtime.deepsort_tracker import DeepSort
from threading import Thread
import queue
import hashlib
from .yolo_model import model, CONF_THRESHOLD, blur_faces
from .feature_extractor import extract_feature
from .geo_fence import MultiGeoFenceManager
from .reid_tracker import PersistentPersonTracker
from .stabilizer import ResponsiveBoxFilter
from .behavior_detector import BehaviorDetector
from .yolo_model import model as yolo_pose_model
from app.video.performance_profiler import get_profiler
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

FACE_BLUR_ENABLED = True
FACE_BLUR_INTERVAL = 1
YOLO_INFERENCE_INTERVAL = 1
BEHAVIOR_ANALYSIS_INTERVAL = 3
DB_SAVE_INTERVAL = 5

_global_uid_hash_map = {}


def hash_id(pid):
    global _global_uid_hash_map

    if pid in _global_uid_hash_map:
        return _global_uid_hash_map[pid]

    hash_str = hashlib.sha256(str(pid).encode()).hexdigest()[:8]
    _global_uid_hash_map[pid] = hash_str
    return hash_str


# ---------------------------------------------------------------------------
# Utility: suppress overlapping confirmed tracks (post-DeepSort NMS)
# ---------------------------------------------------------------------------

def compute_iou(b1, b2):
    """Compute IoU between two boxes in (l, t, r, b) format."""
    xl = max(b1[0], b2[0])
    yt = max(b1[1], b2[1])
    xr = min(b1[2], b2[2])
    yb = min(b1[3], b2[3])
    inter = max(0, xr - xl) * max(0, yb - yt)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-6)


def suppress_overlapping_tracks(tracks, iou_threshold=0.5):
    """
    Remove duplicate DeepSort tracks whose bounding boxes overlap heavily.
    Keeps the track with the lower track_id (older / more stable).
    """
    if len(tracks) <= 1:
        return tracks

    boxes = [tuple(map(int, t.to_ltrb())) for t in tracks]
    suppress = set()

    for i in range(len(boxes)):
        if i in suppress:
            continue
        for j in range(i + 1, len(boxes)):
            if j in suppress:
                continue
            if compute_iou(boxes[i], boxes[j]) > iou_threshold:
                # Keep the older track (lower id); suppress the newer one
                if tracks[i].track_id <= tracks[j].track_id:
                    suppress.add(j)
                else:
                    suppress.add(i)
                    break  # i is suppressed; move to next i

    return [t for idx, t in enumerate(tracks) if idx not in suppress]


# ---------------------------------------------------------------------------

class AnalyticsDBHandler:

    def __init__(self, analytics_db_instance):
        self.db = analytics_db_instance

    def save_person_track(self, camera_id, persistent_id, track_id, bbox, metadata=None):
        if self.db is None:
            return

        try:
            detection_id = self.db.save_detection(
                persistent_id=persistent_id,
                camera_id=camera_id,
                track_id=track_id,
                bbox=bbox,
                confidence=metadata.get('confidence', 0.8) if metadata else 0.8,
                in_geo_fence=metadata.get('in_geo_fence', False) if metadata else False,
                fence_id=metadata.get('fence_id') if metadata else None,
                fence_name=metadata.get('fence_name') if metadata else None,
                frame_id=metadata.get('frame_id', 0) if metadata else 0
            )
            return detection_id
        except Exception as e:
            print(f"[AnalyticsDBHandler] Error saving detection: {e}")
            return None

    def save_behavior_event(self, camera_id, persistent_id, track_id, behavior_type,
                            severity, confidence, description, metadata=None, position=None):
        if self.db is None:
            return

        try:
            detections = self.db.get_detections(
                camera_id=camera_id,
                persistent_id=persistent_id,
                limit=1
            )

            if not detections:
                print(f"[AnalyticsDBHandler] No detection found for UID {persistent_id}")
                return None

            detection_id = detections[0]['id']

            behavior_id = self.db.save_behavior(
                detection_id=detection_id,
                behavior_type=behavior_type,
                severity=severity,
                confidence=confidence,
                description=description,
                metadata=metadata,
                position=position
            )

            self.db.update_person_journey(persistent_id)

            return behavior_id
        except Exception as e:
            print(f"[AnalyticsDBHandler] Error saving behavior: {e}")
            return None

    def deactivate_tracks(self, camera_id, active_ids):
        if self.db is None:
            return

        try:
            detections = self.db.get_detections(camera_id=camera_id, limit=1000)

            all_pids = set(d['persistent_id'] for d in detections)
            inactive_pids = all_pids - set(active_ids)

            for pid in inactive_pids:
                self.db.close_person_journey(pid)
        except Exception as e:
            print(f"[AnalyticsDBHandler] Error deactivating tracks: {e}")


def safe_get_track_confidence(track, default=0.8):
    try:
        if hasattr(track, 'get_det_conf'):
            conf = track.get_det_conf()
            if conf is not None:
                conf_float = float(conf)
                return max(0.0, min(1.0, conf_float))
        return default
    except (TypeError, ValueError, AttributeError):
        return default


BOX_COLOR = (0, 255, 0)
OUTSIDE_COLOR = (0, 165, 255)
LOITERING_COLOR = (0, 165, 255)
RUNNING_COLOR = (0, 0, 255)
VIOLENCE_COLOR = (128, 0, 128)
FALLEN_COLOR = (255, 0, 255)
FIRE_COLOR = (0, 69, 255)
SMOKE_COLOR = (128, 128, 128)
CROWD_COLOR = (0, 255, 255)
PENDING_COLOR = (255, 165, 0)
INITIALIZING_COLOR = (128, 128, 128)


class AlertsSender:
    def __init__(self):
        self.queue = queue.Queue(maxsize=100)
        self.running = True
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.error_count = 0
        self.success_count = 0

    def _worker(self):
        while self.running:
            try:
                payload = self.queue.get(timeout=1.0)
                if payload is None:
                    break

                try:
                    if payload.get('alert_type') == 'behavior':
                        response = requests.post(
                            "http://127.0.0.1:5000/api/alerts/create",
                            json=payload,
                            timeout=2.0
                        )
                    else:
                        response = requests.post(
                            "http://127.0.0.1:5000/api/alerts/yolo-detection",
                            json=payload,
                            timeout=2.0
                        )

                    if response.status_code in [200, 201]:
                        self.success_count += 1
                    else:
                        self.error_count += 1
                        if self.error_count % 10 == 0:
                            print(f"[AlertsSender] ⚠️ HTTP {response.status_code}: {response.text[:100]}")

                except requests.exceptions.ConnectionError:
                    self.error_count += 1
                    if self.error_count == 1:
                        print(f"[AlertsSender] ⚠️ Cannot connect to alerts API. Alerts will be skipped.")
                except requests.exceptions.Timeout:
                    self.error_count += 1
                except Exception as e:
                    self.error_count += 1
                    if self.error_count % 20 == 0:
                        print(f"[AlertsSender] ⚠️ Error: {e}")

                self.queue.task_done()
            except queue.Empty:
                continue

    def send(self, payload):
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            pass

    def get_stats(self):
        return {
            'success': self.success_count,
            'errors': self.error_count,
            'queue_size': self.queue.qsize()
        }

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.thread.join(timeout=2.0)


class VideoStreamManager:
    def __init__(self):
        self.cameras = {}
        self.active_ids = {}

        if analytics_db is not None:
            self.db = AnalyticsDBHandler(analytics_db)
            print("[StreamManager] ✅ Analytics database handler initialized")
        else:
            self.db = None
            print("[StreamManager] ⚠️  Running without database")

        # FIX: Tightened ReID thresholds to reduce duplicate UID spawning
        self.persistent_tracker = PersistentPersonTracker(
            db_handler=self.db if self.db else None,
            similarity_threshold=0.80,        # raised from 0.70
            cross_camera_threshold=0.78,       # raised from 0.72
            confirmation_frames=3,             # raised from 2
            cross_camera_time_window=60.0,
            feature_failure_patience=5,
            use_db_for_matching=True
        )
        print("[StreamManager] ✅ UID tracking initialized (appearance-based only)")

        self.trackers = {}
        self.stabilizers = {}
        self.frame_skip_ratio = 0.0
        self.geo_fence_managers = {}
        self.alerts_sender = AlertsSender()
        self.behavior_detectors = {}

        self.fps_counters = {}

        self.global_permanent_behaviors = {}

    def add_camera(self, camera_id, source):
        self.cameras[camera_id] = {'source': source, 'cap': None, 'active': False}
        self.active_ids[camera_id] = set()
        self.geo_fence_managers[camera_id] = MultiGeoFenceManager()

        # FIX: Tighter DeepSort config to suppress duplicate/overlapping tracks
        self.trackers[camera_id] = DeepSort(
            max_age=15,                  # slightly reduced for faster stale-track cleanup
            n_init=3,
            max_iou_distance=0.4,        # reduced from 0.6 — prevents parallel tracks on same person
            max_cosine_distance=0.3,     # reduced from 0.35 — stricter appearance matching
            nn_budget=150
        )
        self.stabilizers[camera_id] = ResponsiveBoxFilter()
        self.behavior_detectors[camera_id] = BehaviorDetector()
        self.fps_counters[camera_id] = {'count': 0, 'start_time': time.time(), 'fps': 0}

        print(f"[VideoStreamManager] Created tracker, geo-fence manager, and behavior detector for {camera_id}")

    def add_geo_fence(self, camera_id, name, points):
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found")
            return None

        fence_id = self.geo_fence_managers[camera_id].add_fence(name, points)
        if fence_id:
            print(f"[VideoStreamManager] Added geo-fence '{name}' (ID: {fence_id}) to {camera_id}")
        return fence_id

    def remove_geo_fence(self, camera_id, fence_id):
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found")
            return False

        self.geo_fence_managers[camera_id].remove_fence(fence_id)
        print(f"[VideoStreamManager] Removed geo-fence {fence_id} from {camera_id}")
        return True

    def update_geo_fence(self, camera_id, fence_id, points=None, name=None, enabled=None):
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found")
            return False

        success = self.geo_fence_managers[camera_id].update_fence(fence_id, points, name, enabled)
        if success:
            print(f"[VideoStreamManager] Updated geo-fence {fence_id} for {camera_id}")
        return success

    def toggle_geo_fence(self, camera_id, fence_id):
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found")
            return None

        enabled = self.geo_fence_managers[camera_id].toggle_fence(fence_id)
        if enabled is not None:
            status = "enabled" if enabled else "disabled"
            print(f"[VideoStreamManager] Geo-fence {fence_id} {status} for {camera_id}")
        return enabled

    def get_geo_fences(self, camera_id):
        if camera_id in self.geo_fence_managers:
            return self.geo_fence_managers[camera_id].get_all_fences()
        return []

    def load_geo_fences_from_config(self, camera_id, fences_config):
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found")
            return False

        self.geo_fence_managers[camera_id].load_from_config(fences_config)
        print(f"[VideoStreamManager] Loaded geo-fences for {camera_id}")
        return True

    def _convert_to_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _update_fps(self, camera_id):
        counter = self.fps_counters[camera_id]
        counter['count'] += 1

        if counter['count'] % 30 == 0:
            elapsed = time.time() - counter['start_time']
            counter['fps'] = 30 / elapsed
            counter['start_time'] = time.time()

    def start_stream(self, camera_id):
        if camera_id not in self.cameras:
            return False
        camera = self.cameras[camera_id]
        if camera['active']:
            return True
        camera['cap'] = cv2.VideoCapture(camera['source'])
        camera['active'] = True
        socketio.start_background_task(self._stream_frames, camera_id)
        return True

    def stop_stream(self, camera_id):
        if camera_id in self.cameras:
            self.cameras[camera_id]['active'] = False
            if self.cameras[camera_id]['cap']:
                self.cameras[camera_id]['cap'].release()

    def _stream_frames(self, camera_id):
        camera = self.cameras[camera_id]
        cap = camera['cap']
        frame_count = 0
        geo_fence_manager = self.geo_fence_managers[camera_id]

        tracker = self.trackers[camera_id]
        stabilizer = self.stabilizers[camera_id]
        behavior_detector = self.behavior_detectors[camera_id]

        # Initialize profiler
        profiler = get_profiler()
        last_report_frame = 0

        ret, first_frame = cap.read()
        if ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        last_yolo_detections = []
        last_yolo_frame = 0
        last_pose_results = None

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        is_video_file = total_frames > 0
        video_duration_sec = total_frames / fps if fps > 0 else 0

        print(f"[StreamManager] {camera_id} - Video info:")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Duration: {video_duration_sec / 60:.1f} minutes")
        print(f"  Is video file: {is_video_file}")

        while camera['active'] and cap.isOpened():
            with profiler.measure('frame_processing_total'):

                with profiler.measure('frame_read'):
                    ret, frame = cap.read()

                    # Loop video when it ends
                    if not ret:
                        if is_video_file:
                            print(f"[StreamManager] {camera_id} - Video ended, restarting...")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_count = 0
                            ret, frame = cap.read()
                            if not ret:
                                break
                        else:
                            break

                frame_count += 1
                profiler.increment('frames_processed')
                self._update_fps(camera_id)

                if self.frame_skip_ratio > 0 and frame_count % int(1 / (1 - self.frame_skip_ratio)) != 0:
                    socketio.sleep(0.001)
                    continue

                with profiler.measure('frame_resize'):
                    # FIX: only resize very large frames; keep resolution for better detection
                    if frame.shape[1] > 1280:
                        frame = cv2.resize(frame, (1280, 720))
                    elif frame.shape[1] > 960:
                        frame = cv2.resize(frame, (960, 540))

                with profiler.measure('face_blur'):
                    if FACE_BLUR_ENABLED:
                        frame = blur_faces(frame, camera_id=camera_id)

                # YOLO inference with timing
                if frame_count % YOLO_INFERENCE_INTERVAL == 0:
                    with profiler.measure('yolo_inference'):
                        results = yolo_pose_model(
                            frame,
                            conf=CONF_THRESHOLD,
                            iou=0.45,
                            imgsz=960,       # FIX: raised from 640 for better small-person detection
                            verbose=False,
                            device='cpu',
                            half=False
                        )
                        profiler.increment('yolo_runs')

                    with profiler.measure('yolo_postprocess'):
                        detections = []
                        pose_results = results

                        if len(results[0].boxes) > 0:
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                            confs = results[0].boxes.conf.cpu().numpy()
                            classes = results[0].boxes.cls.cpu().numpy()

                            # FIX: Removed redundant cv2.dnn.NMSBoxes — YOLO already applies NMS
                            # internally at iou=0.45. The double-NMS was suppressing valid detections.
                            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                                # FIX: Use a single consistent confidence threshold (0.55)
                                # instead of the original split: CONF_THRESHOLD + hard 0.7 filter
                                if int(cls) == 0 and conf > 0.55:
                                    l, t, r, b = map(int, box)
                                    detections.append(([l, t, r - l, b - t], float(conf), 'person'))

                        last_yolo_detections = detections
                        last_yolo_frame = frame_count
                        last_pose_results = pose_results
                else:
                    detections = last_yolo_detections
                    pose_results = last_pose_results

                with profiler.measure('deepsort_update'):
                    tracks = tracker.update_tracks(detections, frame=frame)
                    tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update < 4]
                    # FIX: suppress overlapping duplicate tracks before any further processing
                    tracks = suppress_overlapping_tracks(tracks, iou_threshold=0.5)
                    profiler.increment('tracks_updated')

                current_frame_ids = set()
                current_track_ids = set()
                inside_count = 0

                person_tracks_for_analysis = []
                all_person_bboxes = []
                current_time = time.time()

                # Prepare ReID detections
                with profiler.measure('reid_prepare'):
                    reid_detections = []
                    for track in tracks:
                        track_id = track.track_id
                        l, t, r, b = map(int, track.to_ltrb())

                        is_inside, fence_id, fence_name = geo_fence_manager.is_person_inside_any((l, t, r, b))

                        if not is_inside:
                            continue

                        inside_count += 1
                        current_track_ids.add(track_id)

                        bbox = [l, t, r - l, b - t]
                        reid_detections.append({
                            'bbox': bbox,
                            'track_id': track_id,
                            'confidence': safe_get_track_confidence(track)
                        })

                # ReID processing
                with profiler.measure('reid_processing'):
                    reid_results = self.persistent_tracker.process_detections(
                        camera_id=camera_id,
                        detections=reid_detections,
                        frame=frame,
                        timestamp=current_time
                    )
                    profiler.increment('reid_processed')

                # Track mapping and database saving
                with profiler.measure('track_mapping_and_db'):
                    track_mapping = {}
                    for result in reid_results:
                        track_id = result['track_id']
                        persistent_id = result['persistent_id']
                        bbox = result['bbox']
                        method = result['method']
                        confidence = result['confidence']

                        current_frame_ids.add(persistent_id)
                        track_mapping[track_id] = persistent_id

                        l, t, w, h = bbox
                        is_inside, fence_id, fence_name = geo_fence_manager.is_person_inside_any((l, t, l + w, t + h))

                        if self.db and frame_count % DB_SAVE_INTERVAL == 0:
                            self.db.save_person_track(
                                camera_id=camera_id,
                                persistent_id=persistent_id,
                                track_id=track_id,
                                bbox=bbox,
                                metadata={
                                    'confidence': confidence,
                                    'in_geo_fence': is_inside,
                                    'fence_id': fence_id,
                                    'fence_name': fence_name,
                                    'frame_id': frame_count,
                                    'method': method
                                }
                            )

                        person_tracks_for_analysis.append({
                            'id': persistent_id,
                            'bbox': bbox
                        })
                        all_person_bboxes.append(bbox)

                # Behavior analysis
                behavior_alerts = []
                if frame_count % BEHAVIOR_ANALYSIS_INTERVAL == 0:
                    with profiler.measure('behavior_analysis'):
                        behavior_alerts = behavior_detector.analyze_frame(
                            person_tracks_for_analysis,
                            frame,
                            current_time,
                            pose_results=pose_results
                        )
                        profiler.increment('behavior_analyzed')

                # Process behavior alerts
                with profiler.measure('alert_processing'):
                    for alert in behavior_alerts:
                        alert_type = alert['type']
                        alert = self._convert_to_json_serializable(alert)

                        if 'person_id' in alert:
                            person_id = alert['person_id']
                            hashed = hash_id(person_id)

                            if alert_type == 'loitering':
                                if person_id not in self.global_permanent_behaviors:
                                    self.global_permanent_behaviors[person_id] = {
                                        'type': 'loitering',
                                        'color': LOITERING_COLOR,
                                        'timestamp': current_time,
                                        'camera_id': camera_id,
                                        'alert': alert
                                    }
                                    print(f"🟡 [LOITERING] UID {person_id} ({hashed}) - Duration: {alert['duration']:.1f}s")

                            elif alert_type in ['running', 'sprinting']:
                                if person_id not in self.global_permanent_behaviors:
                                    self.global_permanent_behaviors[person_id] = {
                                        'type': 'running',
                                        'color': RUNNING_COLOR,
                                        'timestamp': current_time,
                                        'camera_id': camera_id,
                                        'alert': alert
                                    }
                                    speed_info = f"{alert.get('speed_mps', 0):.1f} m/s" if 'speed_mps' in alert else f"{alert.get('speed', 0):.1f} px/s"
                                    print(f"🔴 [RUNNING] UID {person_id} ({hashed}) - Speed: {speed_info}")

                            elif alert_type == 'violence':
                                if person_id not in self.global_permanent_behaviors:
                                    self.global_permanent_behaviors[person_id] = {
                                        'type': 'violence',
                                        'color': VIOLENCE_COLOR,
                                        'timestamp': current_time,
                                        'camera_id': camera_id,
                                        'alert': alert
                                    }
                                    print(f"🟣 [VIOLENCE] UID {person_id} ({hashed}) - Subtype: {alert.get('subtype', 'unknown')}")

                            elif alert_type == 'fallen_person':
                                if person_id not in self.global_permanent_behaviors:
                                    self.global_permanent_behaviors[person_id] = {
                                        'type': 'fallen',
                                        'color': FALLEN_COLOR,
                                        'timestamp': current_time,
                                        'camera_id': camera_id,
                                        'alert': alert
                                    }
                                    print(f"🔴 [FALLEN] UID {person_id} ({hashed}) - Sudden: {alert.get('sudden', False)}")

                            if self.db:
                                self.db.save_behavior_event(
                                    camera_id=camera_id,
                                    persistent_id=person_id,
                                    track_id=None,
                                    behavior_type=alert_type,
                                    severity=alert.get('severity', 'medium'),
                                    confidence=alert.get('confidence', 0.8),
                                    description=f"{alert_type} detected",
                                    metadata={k: v for k, v in alert.items()
                                              if k not in ['type', 'severity', 'description', 'position']},
                                    position=alert.get('position')
                                )

                        elif alert_type == 'crowd_formation':
                            print(f"🟡 [CROWD] {alert['count']} people")
                            position = alert['position']
                            cv2.circle(frame, tuple(position), 80, CROWD_COLOR, 3)
                            cv2.putText(frame, f"CROWD: {alert['count']} people",
                                        (position[0] - 80, position[1] - 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CROWD_COLOR, 2)

                        elif alert_type == 'fire':
                            print(f"🔥 [FIRE] Detected")
                            position = alert['position']
                            cv2.circle(frame, tuple(position), 100, FIRE_COLOR, 4)
                            cv2.putText(frame, "⚠️ FIRE DETECTED",
                                        (position[0] - 100, position[1] - 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, FIRE_COLOR, 3)

                        elif alert_type == 'smoke':
                            print(f"💨 [SMOKE] Detected")
                            position = alert['position']
                            cv2.circle(frame, tuple(position), 100, SMOKE_COLOR, 4)
                            cv2.putText(frame, "⚠️ SMOKE DETECTED",
                                        (position[0] - 100, position[1] - 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, SMOKE_COLOR, 3)

                        profiler.increment(f'alert_{alert_type}')

                        alert_payload = {
                            'alert_type': 'behavior',
                            'type': alert_type,
                            'severity': alert.get('severity', 'medium'),
                            'location': f"{camera_id}",
                            'description': f"{alert_type} detected",
                            'metadata': {
                                'camera_id': camera_id,
                                'frame_id': frame_count,
                                'timestamp': datetime.now().isoformat(),
                                'confidence': alert.get('confidence', 0.8),
                                'person_id': alert.get('person_id'),
                                'position': alert.get('position'),
                                'details': {k: v for k, v in alert.items()
                                            if k not in ['type', 'severity', 'description', 'position']}
                            }
                        }

                        self.alerts_sender.send(alert_payload)

                # Drawing on frame
                with profiler.measure('frame_drawing'):
                    for track in tracks:
                        track_id = track.track_id
                        persistent_id = track_mapping.get(track_id)
                        if persistent_id is None:
                            continue

                        l, t, r, b = map(int, track.to_ltrb())
                        is_inside, fence_id, fence_name = geo_fence_manager.is_person_inside_any((l, t, r, b))
                        if not is_inside:
                            continue

                        hashed = hash_id(persistent_id)
                        color = BOX_COLOR
                        label = f"UID {hashed} [IN:{fence_name}]"

                        if persistent_id in self.global_permanent_behaviors:
                            behavior = self.global_permanent_behaviors[persistent_id]
                            behavior_type = behavior['type']
                            color = behavior['color']

                            if behavior_type == 'loitering':
                                label = f"⚠️ LOITERING UID {hashed} [IN:{fence_name}]"
                            elif behavior_type == 'running':
                                label = f"🏃 RUNNING UID {hashed} [IN:{fence_name}]"
                            elif behavior_type == 'violence':
                                label = f"🥊 VIOLENCE UID {hashed} [IN:{fence_name}]"
                            elif behavior_type == 'fallen':
                                label = f"🤕 FALLEN UID {hashed} [IN:{fence_name}]"

                        cv2.rectangle(frame, (l, t), (r, b), color, 2)
                        cv2.putText(frame, label, (l, t - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Cleanup
                with profiler.measure('cleanup'):
                    if self.db and frame_count % 100 == 0:
                        self.db.deactivate_tracks(camera_id, list(current_frame_ids))

                    all_active_ids = set()
                    for cam_id in self.cameras:
                        all_active_ids.update(self.active_ids.get(cam_id, set()))

                    persons_to_remove = []
                    for person_id in self.global_permanent_behaviors:
                        if person_id not in all_active_ids:
                            persons_to_remove.append(person_id)

                    for person_id in persons_to_remove:
                        del self.global_permanent_behaviors[person_id]

                    behavior_detector.cleanup_old_tracks(current_frame_ids, max_age=30.0)
                    stabilizer.cleanup(current_track_ids)
                    self.persistent_tracker.cleanup_old_tracks(max_age=30.0)

                    self.active_ids[camera_id] = current_frame_ids

                # Frame encoding and emission
                with profiler.measure('frame_encode'):
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')

                with profiler.measure('socketio_emit'):
                    socketio.emit('video_frame', {
                        'camera_id': camera_id,
                        'frame': frame_base64,
                        'timestamp': time.time(),
                        'count': len(current_frame_ids),
                        'inside_count': inside_count,
                        'permanent_behaviors_count': len(self.global_permanent_behaviors),
                        'tracker_stats': self.persistent_tracker.get_statistics(),
                        'fps': self.fps_counters[camera_id]['fps']
                    }, room=f'camera_{camera_id}')

                socketio.sleep(0.03)

                # Print performance report every 300 frames
                if frame_count - last_report_frame >= 300:
                    profiler.print_report()
                    last_report_frame = frame_count

    def get_tracking_statistics(self):
        if hasattr(self, 'persistent_tracker'):
            stats = self.persistent_tracker.get_statistics()
            stats['global_permanent_behaviors'] = len(self.global_permanent_behaviors)
            stats['global_uid_hashes'] = len(_global_uid_hash_map)
            return stats
        return None

    def get_permanent_behaviors(self):
        return {
            pid: {
                'type': behavior['type'],
                'hashed_id': hash_id(pid),
                'camera_id': behavior['camera_id'],
                'timestamp': behavior['timestamp']
            }
            for pid, behavior in self.global_permanent_behaviors.items()
        }

    def clear_permanent_behavior(self, persistent_id):
        if persistent_id in self.global_permanent_behaviors:
            del self.global_permanent_behaviors[persistent_id]
            print(f"🗑️  [ADMIN] Cleared permanent behavior for UID {persistent_id} ({hash_id(persistent_id)})")
            return True
        return False

    def shutdown(self):
        self.alerts_sender.stop()
        print("[StreamManager] ✅ Shutdown complete")


stream_manager = VideoStreamManager()
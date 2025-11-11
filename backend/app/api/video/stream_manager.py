# app/api/video/stream_manager.py
# VERSION: Enhanced with Violence, Fallen Person, Fire/Smoke Detection

import cv2
import base64
import time
import numpy as np
import requests
from datetime import datetime
# ✅ FIXED: Import socketio from extensions, not from app
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

# Analytics Database Integration
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

# --- Configuration ---
FACE_BLUR_ENABLED = True
FACE_BLUR_INTERVAL = 1
YOLO_INFERENCE_INTERVAL = 1
BEHAVIOR_ANALYSIS_INTERVAL = 3
DB_SAVE_INTERVAL = 5


# --- Global UID hash mapping (persists across cameras) ---
_global_uid_hash_map = {}

def hash_id(pid):
    """Convert persistent ID to 8-character hash - CONSISTENT across all cameras"""
    global _global_uid_hash_map
    
    if pid in _global_uid_hash_map:
        return _global_uid_hash_map[pid]
    
    hash_str = hashlib.sha256(str(pid).encode()).hexdigest()[:8]
    _global_uid_hash_map[pid] = hash_str
    return hash_str


class AnalyticsDBHandler:
    """Adapter to make analytics_db compatible with stream manager"""
    
    def __init__(self, analytics_db_instance):
        self.db = analytics_db_instance
    
    def save_person_track(self, camera_id, persistent_id, track_id, bbox, metadata=None):
        """Save person detection to analytics database"""
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
        """Save behavior event to analytics database"""
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
        """Mark inactive person journeys"""
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
    """Safely extract detection confidence from track object."""
    try:
        if hasattr(track, 'get_det_conf'):
            conf = track.get_det_conf()
            if conf is not None:
                conf_float = float(conf)
                return max(0.0, min(1.0, conf_float))
        return default
    except (TypeError, ValueError, AttributeError):
        return default


# Color definitions
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
    """Non-blocking alerts sender using background thread with error handling"""
    def __init__(self):
        self.queue = queue.Queue(maxsize=100)
        self.running = True
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.error_count = 0
        self.success_count = 0
    
    def _worker(self):
        """Background worker that sends alerts"""
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
        """Queue alert payload for sending (non-blocking)"""
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            pass
    
    def get_stats(self):
        """Get sender statistics"""
        return {
            'success': self.success_count,
            'errors': self.error_count,
            'queue_size': self.queue.qsize()
        }
    
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        self.queue.put(None)
        self.thread.join(timeout=2.0)


class VideoStreamManager:
    def __init__(self):
        self.cameras = {}
        self.active_ids = {}
        
        # Database handler (Analytics)
        if analytics_db is not None:
            self.db = AnalyticsDBHandler(analytics_db)
            print("[StreamManager] ✅ Analytics database handler initialized")
        else:
            self.db = None
            print("[StreamManager] ⚠️  Running without database")
        
        # Initialize persistent tracker
        self.persistent_tracker = PersistentPersonTracker(
            db_handler=self.db if self.db else None,
            similarity_threshold=0.70,
            cross_camera_threshold=0.72,
            confirmation_frames=2,
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
        
        # FPS tracking
        self.fps_counters = {}
        
        # GLOBAL PERMANENT BEHAVIOR TRACKING (across all cameras)
        self.global_permanent_behaviors = {}

    def add_camera(self, camera_id, source):
        self.cameras[camera_id] = {'source': source, 'cap': None, 'active': False}
        self.active_ids[camera_id] = set()
        self.geo_fence_managers[camera_id] = MultiGeoFenceManager()
        
        self.trackers[camera_id] = DeepSort(
            max_age=10,
            n_init=3,
            max_iou_distance=0.6,
            max_cosine_distance=0.35,
            nn_budget=200
        )
        self.stabilizers[camera_id] = ResponsiveBoxFilter()
        self.behavior_detectors[camera_id] = BehaviorDetector()
        self.fps_counters[camera_id] = {'count': 0, 'start_time': time.time(), 'fps': 0}
        
        print(f"[VideoStreamManager] Created tracker, geo-fence manager, and behavior detector for {camera_id}")

    def add_geo_fence(self, camera_id, name, points):
        """Add a new geo-fence to a camera"""
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found")
            return None
        
        fence_id = self.geo_fence_managers[camera_id].add_fence(name, points)
        if fence_id:
            print(f"[VideoStreamManager] Added geo-fence '{name}' (ID: {fence_id}) to {camera_id}")
        return fence_id
    
    def remove_geo_fence(self, camera_id, fence_id):
        """Remove a geo-fence from a camera"""
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found")
            return False
        
        self.geo_fence_managers[camera_id].remove_fence(fence_id)
        print(f"[VideoStreamManager] Removed geo-fence {fence_id} from {camera_id}")
        return True
    
    def update_geo_fence(self, camera_id, fence_id, points=None, name=None, enabled=None):
        """Update an existing geo-fence"""
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found")
            return False
        
        success = self.geo_fence_managers[camera_id].update_fence(fence_id, points, name, enabled)
        if success:
            print(f"[VideoStreamManager] Updated geo-fence {fence_id} for {camera_id}")
        return success
    
    def toggle_geo_fence(self, camera_id, fence_id):
        """Toggle geo-fence enabled/disabled"""
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found")
            return None
        
        enabled = self.geo_fence_managers[camera_id].toggle_fence(fence_id)
        if enabled is not None:
            status = "enabled" if enabled else "disabled"
            print(f"[VideoStreamManager] Geo-fence {fence_id} {status} for {camera_id}")
        return enabled
    
    def get_geo_fences(self, camera_id):
        """Get all geo-fences for a camera"""
        if camera_id in self.geo_fence_managers:
            return self.geo_fence_managers[camera_id].get_all_fences()
        return []
    
    def load_geo_fences_from_config(self, camera_id, fences_config):
        """Load multiple geo-fences from configuration"""
        if camera_id not in self.geo_fence_managers:
            print(f"[ERROR] Camera {camera_id} not found")
            return False
        
        self.geo_fence_managers[camera_id].load_from_config(fences_config)
        print(f"[VideoStreamManager] Loaded geo-fences for {camera_id}")
        return True
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON-safe types"""
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
        """Update FPS counter"""
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

        ret, first_frame = cap.read()
        if ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Cache for YOLO detections
        last_yolo_detections = []
        last_yolo_frame = 0

        while camera['active'] and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            self._update_fps(camera_id)
            
            # Skip frames if configured
            if self.frame_skip_ratio > 0 and frame_count % int(1 / (1 - self.frame_skip_ratio)) != 0:
                socketio.sleep(0.001)
                continue

            # Resize frame for performance
            if frame.shape[1] > 960:
                frame = cv2.resize(frame, (960, 540))

            # ===== OPTIMIZED FACE BLUR =====
            if FACE_BLUR_ENABLED:
                frame = blur_faces(frame)

            # ===== YOLO DETECTION =====
            if frame_count % YOLO_INFERENCE_INTERVAL == 0:
                # Run YOLO-Pose detection
                results = yolo_pose_model(frame, conf=CONF_THRESHOLD, iou=0.45, imgsz=640, verbose=False)
                detections = []
                pose_results = results  # Store for behavior analysis

                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()

                    # Apply NMS
                    indices = cv2.dnn.NMSBoxes(
                        [box.tolist() for box in boxes],
                        confs.tolist(),
                        CONF_THRESHOLD,
                        0.35
                    )
                    
                    if len(indices) > 0:
                        indices = indices.flatten()
                        boxes = boxes[indices]
                        confs = confs[indices]
                        classes = classes[indices]

                    for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                        if int(cls) == 0 and conf > 0.7:  # Person class
                            l, t, r, b = map(int, box)
                            detections.append(([l, t, r-l, b-t], float(conf), 'person'))
                
                last_yolo_detections = detections
                last_yolo_frame = frame_count
                last_pose_results = pose_results  # Cache pose results
            else:
                detections = last_yolo_detections
                pose_results = last_pose_results if 'last_pose_results' in locals() else None

            # ===== DEEPSORT TRACKING =====
            tracks = tracker.update_tracks(detections, frame=frame)
            tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update < 4]
            
            current_frame_ids = set()
            current_track_ids = set()
            inside_count = 0
            
            person_tracks_for_analysis = []
            all_person_bboxes = []
            current_time = time.time()

            # ===== FIRST PASS: COLLECT TRACK DATA FOR ANALYSIS =====
            track_mapping = {}
            
            for track_idx, track in enumerate(tracks):
                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                
                # Check geo-fence status
                is_inside, fence_id, fence_name = geo_fence_manager.is_person_inside_any((l, t, r, b))
                
                # Skip if outside geo-fence
                if not is_inside:
                    continue
                
                inside_count += 1
                current_track_ids.add(track_id)

                # Extract crop for ReID
                person_crop = frame[t:b, l:r]
                
                crop_valid = False
                if person_crop.size > 0:
                    crop_h, crop_w = person_crop.shape[:2]
                    crop_area = crop_w * crop_h
                    aspect_ratio = crop_h / max(crop_w, 1)
                    
                    if (crop_area >= 5000 and
                        0.5 <= aspect_ratio <= 4.0 and
                        crop_h >= 64 and crop_w >= 32):
                        
                        brightness = np.mean(person_crop)
                        if 20 < brightness < 235:
                            crop_valid = True
                
                feat = None
                if crop_valid:
                    feat = self.persistent_tracker.extract_feature(person_crop)
                
                # Get or create persistent ID
                persistent_id = self.persistent_tracker.get_or_create_persistent_id(
                    camera_id=camera_id,
                    track_id=track_id,
                    feature=feat,
                    bbox=(l, t, r, b),
                    in_geo_fence=True
                )
                
                # Handle pending confirmation
                if persistent_id is None:
                    track_key = (camera_id, track_id)
                    
                    if track_key in self.persistent_tracker.pending_cross_matches:
                        pending = self.persistent_tracker.pending_cross_matches[track_key]
                        progress = f"{pending['count']}/{self.persistent_tracker.confirmation_frames}"
                        avg_score = np.mean(pending['scores']) if pending['scores'] else 0.0
                        hashed = hash_id(pending['pid'])
                        
                        cv2.rectangle(frame, (l, t), (r, b), PENDING_COLOR, 2)
                        cv2.putText(frame, f"UID {hashed} [PENDING {progress}]", (l, t - 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, PENDING_COLOR, 2)
                        cv2.putText(frame, f"Confirming match ({avg_score:.2f})", (l, t - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, PENDING_COLOR, 1)
                    else:
                        failure_count = self.persistent_tracker.feature_extraction_failures.get(track_key, 0)
                        
                        cv2.rectangle(frame, (l, t), (r, b), INITIALIZING_COLOR, 2)
                        cv2.putText(frame, f"Track {track_id} [INITIALIZING]", (l, t - 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, INITIALIZING_COLOR, 2)
                        cv2.putText(frame, f"Feature extraction: {failure_count}/10", (l, t - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, INITIALIZING_COLOR, 1)
                    
                    continue
                
                current_frame_ids.add(persistent_id)
                track_mapping[track_id] = persistent_id

                bbox = [l, t, r - l, b - t]
                
                # Save to database
                if self.db and frame_count % DB_SAVE_INTERVAL == 0:
                    self.db.save_person_track(
                        camera_id=camera_id,
                        persistent_id=persistent_id,
                        track_id=track_id,
                        bbox=bbox,
                        metadata={
                            'confidence': safe_get_track_confidence(track),
                            'in_geo_fence': is_inside,
                            'fence_id': fence_id,
                            'fence_name': fence_name,
                            'frame_id': frame_count
                        }
                    )

                # Collect data for behavior analysis
                person_tracks_for_analysis.append({
                    'id': persistent_id,
                    'bbox': bbox
                })
                all_person_bboxes.append(bbox)

            # ===== BEHAVIOR ANALYSIS =====
                behavior_alerts = []
                if frame_count % BEHAVIOR_ANALYSIS_INTERVAL == 0:
                    # Pass pose results to behavior detector
                    behavior_alerts = behavior_detector.analyze_frame(
                        person_tracks_for_analysis,
                        frame,
                        current_time,
                        pose_results=pose_results  if 'pose_results' in locals() else None
                    )
                
                # Process behavior alerts and update permanent behaviors
                for alert in behavior_alerts:
                    alert_type = alert['type']
                    alert = self._convert_to_json_serializable(alert)
                    
                    # Handle person-specific behaviors
                    if 'person_id' in alert:
                        person_id = alert['person_id']
                        hashed = hash_id(person_id)
                        
                        # Store permanent behaviors
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
                        
                        elif alert_type == 'running':
                            if person_id not in self.global_permanent_behaviors:
                                self.global_permanent_behaviors[person_id] = {
                                    'type': 'running',
                                    'color': RUNNING_COLOR,
                                    'timestamp': current_time,
                                    'camera_id': camera_id,
                                    'alert': alert
                                }
                                print(f"🔴 [RUNNING] UID {person_id} ({hashed}) - Velocity: {alert['velocity']:.1f} px/s")
                        
                        elif alert_type == 'violence':
                            if person_id not in self.global_permanent_behaviors:
                                self.global_permanent_behaviors[person_id] = {
                                    'type': 'violence',
                                    'color': VIOLENCE_COLOR,
                                    'timestamp': current_time,
                                    'camera_id': camera_id,
                                    'alert': alert
                                }
                                print(f"🟣 [VIOLENCE] UID {person_id} ({hashed}) - {alert['description']}")
                        
                        elif alert_type == 'fallen':
                            if person_id not in self.global_permanent_behaviors:
                                self.global_permanent_behaviors[person_id] = {
                                    'type': 'fallen',
                                    'color': FALLEN_COLOR,
                                    'timestamp': current_time,
                                    'camera_id': camera_id,
                                    'alert': alert
                                }
                                print(f"🔴 [FALLEN] UID {person_id} ({hashed}) - {alert['description']}")
                        
                        # Save behavior to database
                        if self.db:
                            self.db.save_behavior_event(
                                camera_id=camera_id,
                                persistent_id=person_id,
                                track_id=None,
                                behavior_type=alert_type,
                                severity=alert['severity'],
                                confidence=alert.get('confidence', 0.8),
                                description=alert.get('description', f"{alert_type} detected"),
                                metadata={k: v for k, v in alert.items() 
                                        if k not in ['type', 'severity', 'description', 'position']},
                                position=alert.get('position')
                            )
                    
                    # Handle non-person alerts (crowd, fire, smoke)
                    elif alert_type == 'crowd':
                        print(f"🟡 [CROWD] {alert['person_count']} people in zone {alert['zone_id']}")
                        
                        # Draw crowd circles
                        zone_center = alert['zone_center']
                        cv2.circle(frame, tuple(zone_center), 80, CROWD_COLOR, 3)
                        cv2.putText(frame, f"CROWD: {alert['person_count']} people", 
                                   (zone_center[0] - 80, zone_center[1] - 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, CROWD_COLOR, 2)
                    
                    elif alert_type == 'fire':
                        print(f"🔥 [FIRE] {alert['description']}")
                        
                        # Draw fire warning
                        position = alert['position']
                        cv2.circle(frame, tuple(position), 100, FIRE_COLOR, 4)
                        cv2.putText(frame, "⚠️ FIRE DETECTED", 
                                   (position[0] - 100, position[1] - 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, FIRE_COLOR, 3)
                    
                    elif alert_type == 'smoke':
                        print(f"💨 [SMOKE] {alert['description']}")
                        
                        # Draw smoke warning
                        position = alert['position']
                        cv2.circle(frame, tuple(position), 100, SMOKE_COLOR, 4)
                        cv2.putText(frame, "⚠️ SMOKE DETECTED", 
                                   (position[0] - 100, position[1] - 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, SMOKE_COLOR, 3)
                    
                    # Send alert
                    alert_payload = {
                        'alert_type': 'behavior',
                        'type': alert_type,
                        'severity': alert['severity'],
                        'location': f"{camera_id}",
                        'description': alert.get('description', f"{alert_type} detected"),
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

            # ===== SECOND PASS: DRAW BOXES WITH CORRECT COLORS =====
            for track in tracks:
                track_id = track.track_id
                
                # Get persistent ID from mapping
                persistent_id = track_mapping.get(track_id)
                if persistent_id is None:
                    continue
                
                l, t, r, b = map(int, track.to_ltrb())
                
                # Get geo-fence info
                is_inside, fence_id, fence_name = geo_fence_manager.is_person_inside_any((l, t, r, b))
                if not is_inside:
                    continue
                
                hashed = hash_id(persistent_id)
                
                # ===== DETERMINE COLOR & LABEL =====
                color = BOX_COLOR  # Default green
                label = f"UID {hashed} [IN:{fence_name}]"
                
                # Check permanent behaviors
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
                
                # Draw box and label
                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.putText(frame, label, (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ===== DEACTIVATE TRACKS IN DATABASE =====
            if self.db and frame_count % 100 == 0:
                self.db.deactivate_tracks(camera_id, list(current_frame_ids))

            # ===== GLOBAL CLEANUP =====
            all_active_ids = set()
            for cam_id in self.cameras:
                all_active_ids.update(self.active_ids.get(cam_id, set()))
            
            persons_to_remove = []
            for person_id in self.global_permanent_behaviors:
                if person_id not in all_active_ids:
                    persons_to_remove.append(person_id)
            
            for person_id in persons_to_remove:
                del self.global_permanent_behaviors[person_id]
                print(f"🗑️  [CLEANUP] UID {person_id} ({hash_id(person_id)}) removed - not seen in any camera")

            # ===== CLEANUP =====
            behavior_detector.cleanup_old_tracks(current_frame_ids, max_age=30.0)
            stabilizer.cleanup(current_track_ids)
            self.persistent_tracker.cleanup_old_tracks(camera_id, current_track_ids)
            self.active_ids[camera_id] = current_frame_ids

            # ===== SEND DETECTIONS TO ALERTS API =====
            if len(tracks) > 0 and frame_count % 10 == 0:
                detections_payload = {
                    "camera_id": camera_id,
                    "frame_id": frame_count,
                    "timestamp": datetime.now().isoformat(),
                    "detections": []
                }
                
                for track in tracks:
                    track_id = track.track_id
                    l, t, r, b = map(int, track.to_ltrb())
                    centroid = ((l + r) // 2, (t + b) // 2)
                    confidence = safe_get_track_confidence(track, default=0.8)
                    
                    detections_payload["detections"].append({
                        "class": "person",
                        "track_id": int(track_id),
                        "centroid": [int(centroid[0]), int(centroid[1])],
                        "bbox": [int(l), int(t), int(r - l), int(b - t)],
                        "confidence": float(confidence)
                    })
                
                self.alerts_sender.send(detections_payload)

            # ===== UPDATE HOURLY STATS =====
            if self.db and frame_count % 300 == 0:
                try:
                    analytics_db.update_hourly_stats(camera_id, datetime.utcnow())
                except Exception as e:
                    print(f"[StreamManager] Error updating hourly stats: {e}")

            # ===== STATS LOGGING =====
            if frame_count % 100 == 0:
                stats = self.persistent_tracker.get_statistics()
                fps = self.fps_counters[camera_id]['fps']
                permanent_count = len(self.global_permanent_behaviors)
                print(f"[StreamManager] {camera_id} - FPS: {fps:.1f}, Active UIDs: {stats['active_uids_total']}, "
                      f"Pending: {stats['pending_cross_matches']}, Next UID: {stats['next_id']}, "
                      f"Global Permanent Behaviors: {permanent_count}")

            # ===== ENCODE AND SEND FRAME =====
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

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

    def get_tracking_statistics(self):
        """Get tracking statistics"""
        if hasattr(self, 'persistent_tracker'):
            stats = self.persistent_tracker.get_statistics()
            stats['global_permanent_behaviors'] = len(self.global_permanent_behaviors)
            stats['global_uid_hashes'] = len(_global_uid_hash_map)
            return stats
        return None
    
    def get_permanent_behaviors(self):
        """Get all permanent behaviors across all cameras"""
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
        """Manually clear a person's permanent behavior (for admin use)"""
        if persistent_id in self.global_permanent_behaviors:
            del self.global_permanent_behaviors[persistent_id]
            print(f"🗑️  [ADMIN] Cleared permanent behavior for UID {persistent_id} ({hash_id(persistent_id)})")
            return True
        return False

    def shutdown(self):
        """Cleanup when shutting down"""
        self.alerts_sender.stop()
        print("[StreamManager] ✅ Shutdown complete")


stream_manager = VideoStreamManager()
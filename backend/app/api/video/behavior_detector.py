"""
Enhanced Behavior Detection System - OPTIMIZED FOR 8-12 FPS
- Fire/Smoke: YOLO Object Detection (trained model) with color fallback
- Violence: YOLO-Pose Skeleton Analysis with multi-frame validation
- Loitering, Running, Fallen: Movement-based with FPS-aware thresholds
- Crowd: Spatial clustering
- Face Blur Integration: Removes blur from violent persons

IMPROVEMENTS:
- FPS-aware detection thresholds
- Multi-frame confirmation to reduce false positives
- Velocity spike filtering for tracking errors
- Consistency checks for violence detection
- Adaptive thresholds based on measured FPS
"""

import numpy as np
import cv2
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
import math

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[BehaviorDetector] ⚠️ Ultralytics not available - fire/smoke detection will use color-based fallback")

# Import face blur violence tracker
try:
    from .yolo_model import register_violence_event, reset_violence_state
    FACE_BLUR_INTEGRATION = True
    print("[BehaviorDetector] ✅ Face blur integration enabled")
except ImportError:
    FACE_BLUR_INTEGRATION = False
    print("[BehaviorDetector] ⚠️ Face blur integration not available")


class EnhancedBehaviorDetector:
    """
    Enhanced behavior detection with YOLO models and face blur integration
    OPTIMIZED FOR LOW FPS (8-12 FPS) CAMERAS
    """
    
    def __init__(self, target_fps=10.0):
        """
        Initialize detector with FPS-aware thresholds
        
        Args:
            target_fps: Expected camera FPS (default 10.0 for typical CCTV)
        """
        # Track person movements and positions over time
        self.person_tracks = defaultdict(lambda: {
            'positions': deque(maxlen=150),
            'timestamps': deque(maxlen=150),
            'velocities': deque(maxlen=50),  # Increased for smoothing
            'accelerations': deque(maxlen=30),
            'bbox_sizes': deque(maxlen=30),
            'aspect_ratios': deque(maxlen=30),
            'keypoints_history': deque(maxlen=50),  # Increased for better analysis
            'first_seen': None,
            'last_seen': None,
            'total_distance': 0.0,
            'time_stationary': 0.0,
            'loitering_alerted': False,
            'running_alerted': False,
            'fallen_alerted': False,
            'violence_alerted': False,
            'zone_entry_time': None,
            'last_zone': None,
            'last_bbox': None,
            'was_standing': True,
            'fall_timestamp': None,
            'is_violent': False,
            'violence_start_time': None,
            'violence_frame_count': 0,  # NEW: Count violent frames
            'running_frame_count': 0,  # NEW: Count running frames
        })
        
        # FPS tracking
        self.fps_history = deque(maxlen=100)
        self.estimated_fps = target_fps
        self.target_fps = target_fps
        
        # Configuration thresholds (FPS-AWARE)
        self.config = {
            # Loitering detection (TIME-BASED - works at any FPS)
            'loitering_time_threshold': 45.0,  # seconds
            'loitering_distance_threshold': 150,  # pixels
            'loitering_min_frames': int(target_fps * 5),  # 5 seconds of data
            'stationary_velocity_threshold': 3.0,  # px/s
            
            # Running detection (FRAME + TIME BASED)
            'running_velocity_threshold': 30.0,  # px/s (adjust per camera distance)
            'running_duration_threshold': 2.5,  # seconds of sustained running
            'running_min_frames': int(target_fps * 2.5),  # 2.5 seconds
            'running_confirmation_ratio': 0.80,  # 80% of frames must show running
            'max_reasonable_velocity': 250.0,  # px/s (filter tracking errors)
            
            # Violence detection (POSE-BASED with MULTI-FRAME VALIDATION)
            'violence_arm_movement_threshold': 40.0,  # px/s arm velocity
            'violence_proximity_threshold': 100,  # pixels
            'violence_pose_variance_threshold': 150.0,
            'violence_duration_threshold': 2.0,  # seconds
            'violence_min_frames': int(target_fps * 2.0),  # 2 seconds minimum
            'violence_confirmation_ratio': 0.70,  # 70% of frames must show violence
            'violence_nearby_threshold': 25.0,  # nearby person arm velocity
            
            # Fallen person detection
            'standing_aspect_ratio': 1.5,  # height/width when standing
            'fallen_aspect_ratio': 0.7,  # height/width when fallen
            'fall_y_change_threshold': 80,  # pixels
            'fallen_duration_threshold': 2.0,  # seconds (reduced for faster alert)
            'fallen_velocity_threshold': 2.0,  # px/s
            'fallen_min_frames': int(target_fps * 1.0),  # 1 second
            
            # Fire/Smoke detection (YOLO-BASED)
            'fire_confidence_threshold': 0.50,
            'smoke_confidence_threshold': 0.40,
            'fire_persistence_threshold': 2.0,  # seconds (reduced)
            'fire_min_frames': int(target_fps * 2.0),  # 2 seconds
            
            # Crowd density
            'crowd_density_threshold': 12,  # people in zone
            'crowd_area_size': 200,  # pixels
            'high_density_threshold': 20,  # critical crowd level
            'crowd_min_frames': int(target_fps * 3.0),  # 3 seconds of sustained crowd
            
            # Alert cooldown
            'alert_cooldown': 30.0  # seconds between repeat alerts
        }
        
        # Crowd tracking
        self.crowd_zones = defaultdict(lambda: {
            'person_count': 0,
            'last_alert': 0,
            'density_history': deque(maxlen=50),
            'high_density_frames': 0
        })
        
        # Fire/Smoke tracking
        self.fire_regions = defaultdict(lambda: {
            'first_detected': None,
            'last_detected': None,
            'confidence_history': deque(maxlen=50),
            'bbox_history': deque(maxlen=20),
            'alerted': False,
            'detection_frames': 0
        })
        
        # Alert history for cooldown management
        self.alert_history = defaultdict(lambda: {
            'loitering': 0,
            'running': 0,
            'violence': 0,
            'fallen': 0,
            'fire': 0,
            'smoke': 0
        })
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'violence_false_positive_filtered': 0,
            'running_false_positive_filtered': 0,
            'velocity_spikes_filtered': 0
        }
        
        # Initialize YOLO models
        self._initialize_yolo_models()
        
        print(f"[BehaviorDetector] ✅ Initialized for {target_fps} FPS")
        print(f"[BehaviorDetector] Violence requires {self.config['violence_min_frames']} frames ({self.config['violence_duration_threshold']}s)")
        print(f"[BehaviorDetector] Running requires {self.config['running_min_frames']} frames ({self.config['running_duration_threshold']}s)")
    
    def _initialize_yolo_models(self):
        """Initialize YOLO models for fire/smoke detection"""
        self.fire_smoke_model = None
        
        if not YOLO_AVAILABLE:
            print("[BehaviorDetector] ⚠️ YOLO not available - using fallback color detection")
            return
        
        try:
            model_paths = [
                'models/fire-smoke-yolov8n.pt',
                'models/fire_smoke.pt',
            ]
            
            for model_path in model_paths:
                try:
                    self.fire_smoke_model = YOLO(model_path)
                    self.fire_smoke_model.fuse()
                    print(f"[BehaviorDetector] ✅ Fire/Smoke model loaded: {model_path}")
                    break
                except:
                    continue
            
            if self.fire_smoke_model is None:
                print("[BehaviorDetector] ⚠️ No fire/smoke model found - using color detection")
        
        except Exception as e:
            print(f"[BehaviorDetector] ⚠️ Error loading fire/smoke model: {e}")
            self.fire_smoke_model = None
    
    def update_fps_estimate(self, time_delta):
        """
        Update FPS estimation based on frame timing
        
        Args:
            time_delta: Time between frames in seconds
        """
        if time_delta > 0 and time_delta < 1.0:  # Sanity check
            instant_fps = 1.0 / time_delta
            self.fps_history.append(instant_fps)
            
            if len(self.fps_history) >= 30:
                # Use median to filter outliers
                self.estimated_fps = np.median(list(self.fps_history))
                
                # Log FPS changes
                if abs(self.estimated_fps - self.target_fps) > 2.0:
                    if self.stats['total_frames_processed'] % 100 == 0:
                        print(f"[BehaviorDetector] 📊 Measured FPS: {self.estimated_fps:.1f} (target: {self.target_fps})")
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_velocity(self, pos1, pos2, time_delta):
        """
        Calculate velocity with spike filtering for tracking errors
        
        Args:
            pos1, pos2: Positions
            time_delta: Time between positions
            
        Returns:
            Filtered velocity in pixels per second
        """
        if time_delta <= 0:
            return 0.0
        
        distance = self.calculate_distance(pos1, pos2)
        raw_velocity = distance / time_delta
        
        # Filter unrealistic velocities (tracking errors at low FPS)
        max_velocity = self.config['max_reasonable_velocity']
        
        if raw_velocity > max_velocity:
            self.stats['velocity_spikes_filtered'] += 1
            return 0.0  # Ignore this velocity measurement
        
        return raw_velocity
    
    def calculate_aspect_ratio(self, bbox):
        """Calculate aspect ratio (height/width) of bounding box"""
        width = bbox[2]
        height = bbox[3]
        if width == 0:
            return 0
        return height / width
    
    def calculate_bbox_area(self, bbox):
        """Calculate area of bounding box"""
        return bbox[2] * bbox[3]
    
    def extract_keypoints(self, pose_result):
        """Extract keypoints from YOLO-Pose result"""
        if pose_result is None or len(pose_result) == 0:
            return None
        
        try:
            keypoints = pose_result[0].keypoints.xy.cpu().numpy()[0]
            
            return {
                'nose': keypoints[0],
                'left_shoulder': keypoints[5],
                'right_shoulder': keypoints[6],
                'left_elbow': keypoints[7],
                'right_elbow': keypoints[8],
                'left_wrist': keypoints[9],
                'right_wrist': keypoints[10],
                'left_hip': keypoints[11],
                'right_hip': keypoints[12]
            }
        except Exception as e:
            return None
    
    def calculate_arm_velocity(self, current_kp, prev_kp, time_delta):
        """Calculate arm movement velocity"""
        if current_kp is None or prev_kp is None or time_delta <= 0:
            return 0.0
        
        try:
            left_wrist_dist = self.calculate_distance(
                current_kp['left_wrist'], 
                prev_kp['left_wrist']
            )
            right_wrist_dist = self.calculate_distance(
                current_kp['right_wrist'], 
                prev_kp['right_wrist']
            )
            
            left_elbow_dist = self.calculate_distance(
                current_kp['left_elbow'], 
                prev_kp['left_elbow']
            )
            right_elbow_dist = self.calculate_distance(
                current_kp['right_elbow'], 
                prev_kp['right_elbow']
            )
            
            max_movement = max(left_wrist_dist, right_wrist_dist, 
                             left_elbow_dist, right_elbow_dist)
            
            velocity = max_movement / time_delta
            
            # Filter unrealistic arm movements
            if velocity > 300.0:  # Unrealistic arm speed
                return 0.0
            
            return velocity
        except:
            return 0.0
    
    def update_person(self, person_id, bbox, timestamp, keypoints=None):
        """Update person tracking data with FPS tracking"""
        track = self.person_tracks[person_id]
        
        centroid = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        
        if track['first_seen'] is None:
            track['first_seen'] = timestamp
            track['zone_entry_time'] = timestamp
        
        track['last_seen'] = timestamp
        track['last_bbox'] = bbox
        
        # Update FPS estimate
        if len(track['timestamps']) >= 1:
            prev_time = track['timestamps'][-1]
            time_delta = timestamp - prev_time
            self.update_fps_estimate(time_delta)
        
        track['positions'].append(centroid)
        track['timestamps'].append(timestamp)
        
        aspect_ratio = self.calculate_aspect_ratio(bbox)
        bbox_area = self.calculate_bbox_area(bbox)
        
        track['aspect_ratios'].append(aspect_ratio)
        track['bbox_sizes'].append(bbox_area)
        
        # Calculate velocity with filtering
        if len(track['positions']) >= 2:
            prev_pos = track['positions'][-2]
            prev_time = track['timestamps'][-2]
            time_delta = timestamp - prev_time
            
            if time_delta > 0:
                velocity = self.calculate_velocity(centroid, prev_pos, time_delta)
                
                # Only add non-zero velocities (filtered tracking errors)
                if velocity > 0:
                    track['velocities'].append(velocity)
                
                if len(track['velocities']) >= 2:
                    prev_velocity = track['velocities'][-2]
                    acceleration = (velocity - prev_velocity) / time_delta
                    track['accelerations'].append(acceleration)
                
                distance = self.calculate_distance(centroid, prev_pos)
                track['total_distance'] += distance
                
                # Track stationary time
                if velocity < self.config['stationary_velocity_threshold']:
                    track['time_stationary'] += time_delta
                else:
                    track['time_stationary'] = 0
    
    def detect_loitering(self, person_id, current_time):
        """
        Detect loitering with time-based threshold (FPS-independent)
        """
        track = self.person_tracks[person_id]
        
        # Cooldown check
        if current_time - self.alert_history[person_id]['loitering'] < self.config['alert_cooldown']:
            return None
        
        # Need minimum data
        min_frames = self.config['loitering_min_frames']
        if len(track['positions']) < min_frames:
            return None
        
        # Time-based threshold
        time_in_area = current_time - track['first_seen']
        
        if time_in_area < self.config['loitering_time_threshold']:
            return None
        
        # Check movement range
        positions = list(track['positions'])
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        movement_range = max(
            max(x_coords) - min(x_coords),
            max(y_coords) - min(y_coords)
        )
        
        if movement_range < self.config['loitering_distance_threshold']:
            if len(track['velocities']) > 0:
                avg_velocity = np.mean(list(track['velocities']))
                
                if avg_velocity < self.config['stationary_velocity_threshold'] * 1.5:
                    self.alert_history[person_id]['loitering'] = current_time
                    
                    return {
                        'type': 'loitering',
                        'severity': 'medium',
                        'person_id': int(person_id),
                        'duration': float(time_in_area),
                        'movement_range': float(movement_range),
                        'avg_velocity': float(avg_velocity),
                        'position': [int(positions[-1][0]), int(positions[-1][1])],
                        'confidence': float(min(0.95, 0.6 + (time_in_area / 30.0))),
                        'description': f'Person loitering for {time_in_area:.1f}s (moved only {movement_range:.0f}px)'
                    }
        
        return None
    
    def detect_running(self, person_id, current_time):
        """
        Detect running with multi-frame confirmation to reduce false positives
        """
        track = self.person_tracks[person_id]
        
        # Cooldown check
        if current_time - self.alert_history[person_id]['running'] < self.config['alert_cooldown']:
            return None
        
        # Need minimum frames
        min_frames = self.config['running_min_frames']
        if len(track['velocities']) < min_frames:
            return None
        
        # Analyze recent velocities
        recent_velocities = list(track['velocities'])[-min_frames:]
        
        if len(recent_velocities) == 0:
            return None
        
        avg_velocity = np.mean(recent_velocities)
        threshold = self.config['running_velocity_threshold']
        
        # Count frames above running threshold
        high_velocity_frames = sum(1 for v in recent_velocities if v > threshold * 0.85)
        confirmation_ratio = high_velocity_frames / len(recent_velocities)
        
        # Require sustained high velocity (not just brief spikes)
        required_ratio = self.config['running_confirmation_ratio']
        
        if avg_velocity > threshold and confirmation_ratio >= required_ratio:
            # Additional check: ensure sustained duration
            if len(track['timestamps']) >= 2:
                duration = track['timestamps'][-1] - track['timestamps'][-min_frames]
                
                if duration >= self.config['running_duration_threshold']:
                    self.alert_history[person_id]['running'] = current_time
                    
                    return {
                        'type': 'running',
                        'severity': 'high',
                        'person_id': int(person_id),
                        'velocity': float(avg_velocity),
                        'max_velocity': float(max(recent_velocities)),
                        'duration': float(duration),
                        'confirmation_ratio': float(confirmation_ratio),
                        'position': [int(list(track['positions'])[-1][0]), int(list(track['positions'])[-1][1])],
                        'confidence': float(min(0.95, 0.7 + (confirmation_ratio * 0.2))),
                        'description': f'Person running at {avg_velocity:.1f}px/s for {duration:.1f}s ({confirmation_ratio*100:.0f}% confirmation)'
                    }
        else:
            # Track filtered false positives
            if avg_velocity > threshold:
                self.stats['running_false_positive_filtered'] += 1
        
        return None
    
    def detect_violence_pose(self, person_id, current_time, all_person_positions, keypoints=None):
        """
        Detect violent behavior using POSE analysis with multi-frame validation
        
        CRITICAL: When violence is detected, face blur is DISABLED for involved persons
        """
        track = self.person_tracks[person_id]
        
        # Cooldown check
        if current_time - self.alert_history[person_id]['violence'] < self.config['alert_cooldown']:
            return None
        
        # Store keypoints history
        if keypoints is not None:
            track['keypoints_history'].append({
                'keypoints': keypoints,
                'timestamp': current_time
            })
        
        # Need minimum frames for reliable detection
        min_frames = self.config['violence_min_frames']
        if len(track['keypoints_history']) < min_frames:
            return None
        
        recent_poses = list(track['keypoints_history'])[-min_frames:]
        
        # Analyze arm movements across multiple frames
        arm_velocities = []
        violent_frames = 0
        
        for i in range(1, len(recent_poses)):
            prev_kp = recent_poses[i-1]['keypoints']
            curr_kp = recent_poses[i]['keypoints']
            time_delta = recent_poses[i]['timestamp'] - recent_poses[i-1]['timestamp']
            
            arm_vel = self.calculate_arm_velocity(curr_kp, prev_kp, time_delta)
            arm_velocities.append(arm_vel)
            
            # Count frames with high arm velocity
            if arm_vel > self.config['violence_arm_movement_threshold']:
                violent_frames += 1
        
        if len(arm_velocities) == 0:
            return None
        
        # Calculate violence metrics
        max_arm_velocity = max(arm_velocities)
        avg_arm_velocity = np.mean(arm_velocities)
        arm_velocity_variance = np.var(arm_velocities)
        
        # Confirmation ratio: what % of frames show violent movement?
        confirmation_ratio = violent_frames / len(arm_velocities)
        required_ratio = self.config['violence_confirmation_ratio']
        
        # Check for nearby persons
        current_pos = list(track['positions'])[-1] if track['positions'] else None
        if current_pos is None:
            return None
        
        nearby_persons = []
        for other_id, other_pos in all_person_positions.items():
            if other_id != person_id:
                distance = self.calculate_distance(current_pos, other_pos)
                if distance < self.config['violence_proximity_threshold']:
                    nearby_persons.append(other_id)
        
        # Check if nearby persons also show violent behavior
        nearby_violent = False
        if nearby_persons:
            for nearby_id in nearby_persons:
                nearby_track = self.person_tracks[nearby_id]
                if len(nearby_track['keypoints_history']) >= 5:
                    nearby_poses = list(nearby_track['keypoints_history'])[-5:]
                    nearby_arm_vels = []
                    
                    for i in range(1, len(nearby_poses)):
                        prev_kp = nearby_poses[i-1]['keypoints']
                        curr_kp = nearby_poses[i]['keypoints']
                        t_delta = nearby_poses[i]['timestamp'] - nearby_poses[i-1]['timestamp']
                        nearby_arm_vels.append(self.calculate_arm_velocity(curr_kp, prev_kp, t_delta))
                    
                    if len(nearby_arm_vels) > 0:
                        nearby_max_vel = max(nearby_arm_vels)
                        if nearby_max_vel > self.config['violence_nearby_threshold']:
                            nearby_violent = True
                            break
        
        # VIOLENCE DETECTION CRITERIA (ALL MUST BE TRUE):
        # 1. Rapid arm movement (peak velocity)
        # 2. Sustained over multiple frames (confirmation ratio)
        # 3. Erratic movement pattern (variance)
        # 4. Person nearby (proximity)
        # 5. Nearby person also shows violent behavior
        
        rapid_arm_movement = max_arm_velocity > self.config['violence_arm_movement_threshold']
        sustained_violence = confirmation_ratio >= required_ratio
        erratic_movement = arm_velocity_variance > self.config['violence_pose_variance_threshold']
        has_nearby = len(nearby_persons) > 0
        
        if rapid_arm_movement and sustained_violence and erratic_movement and has_nearby and nearby_violent:
            # Additional duration check
            duration = recent_poses[-1]['timestamp'] - recent_poses[0]['timestamp']
            
            if duration >= self.config['violence_duration_threshold']:
                self.alert_history[person_id]['violence'] = current_time
                
                # Mark person as violent
                track['is_violent'] = True
                track['violence_start_time'] = current_time
                
                # Mark nearby persons as violent
                for nearby_id in nearby_persons:
                    nearby_track = self.person_tracks[nearby_id]
                    nearby_track['is_violent'] = True
                    nearby_track['violence_start_time'] = current_time
                
                # Collect all violent person IDs
                violent_person_ids = [person_id] + nearby_persons
                
                # CRITICAL: Disable face blur for violent persons
                if FACE_BLUR_INTEGRATION:
                    try:
                        register_violence_event(violent_person_ids, zone_id=f"violence_{person_id}")
                        print(f"[BehaviorDetector] 🚨 Violence detected - Face blur REMOVED for persons: {violent_person_ids}")
                    except Exception as e:
                        print(f"[BehaviorDetector] ⚠️ Failed to disable face blur: {e}")
                
                return {
                    'type': 'violence',
                    'severity': 'critical',
                    'person_id': int(person_id),
                    'max_arm_velocity': float(max_arm_velocity),
                    'avg_arm_velocity': float(avg_arm_velocity),
                    'arm_variance': float(arm_velocity_variance),
                    'confirmation_ratio': float(confirmation_ratio),
                    'duration': float(duration),
                    'nearby_persons': [int(p) for p in nearby_persons],
                    'position': [int(current_pos[0]), int(current_pos[1])],
                    'confidence': float(min(0.92, 0.65 + (confirmation_ratio * 0.25))),
                    'description': f'Physical altercation detected ({confirmation_ratio*100:.0f}% confirmed over {duration:.1f}s) with {len(nearby_persons)} person(s)',
                    'detection_method': 'yolo_pose_multi_frame',
                    'face_blur_disabled': FACE_BLUR_INTEGRATION,
                    'violent_person_ids': [int(p) for p in violent_person_ids]
                }
        else:
            # Track filtered false positives
            if rapid_arm_movement and has_nearby:
                self.stats['violence_false_positive_filtered'] += 1
        
        return None
    
    def detect_fallen_person(self, person_id, current_time):
        """
        Detect fallen person with reduced false positives
        """
        track = self.person_tracks[person_id]
        
        # Cooldown check
        if current_time - self.alert_history[person_id]['fallen'] < self.config['alert_cooldown']:
            return None
        
        # Need minimum data
        min_frames = self.config['fallen_min_frames']
        if len(track['aspect_ratios']) < min_frames or len(track['positions']) < min_frames:
            return None
        
        current_ratio = track['aspect_ratios'][-1]
        prev_ratios = list(track['aspect_ratios'])[-min_frames:-1]
        
        # Check if person was standing before
        was_standing = any(r > self.config['standing_aspect_ratio'] for r in prev_ratios[-5:])
        is_horizontal = current_ratio < self.config['fallen_aspect_ratio']
        
        if not (was_standing and is_horizontal):
            return None
        
        # Check for significant Y-position change (falling motion)
        if len(track['positions']) >= 5:
            prev_pos = track['positions'][-5]
            current_pos = track['positions'][-1]
            y_change = current_pos[1] - prev_pos[1]
            
            if y_change > self.config['fall_y_change_threshold']:
                # Start tracking fall
                if track['fall_timestamp'] is None:
                    track['fall_timestamp'] = current_time
                
                time_horizontal = current_time - track['fall_timestamp']
                
                # Alert if motionless for duration threshold
                if time_horizontal >= self.config['fallen_duration_threshold']:
                    if len(track['velocities']) > 0:
                        recent_velocity = np.mean(list(track['velocities'])[-min_frames:])
                        
                        if recent_velocity < self.config['fallen_velocity_threshold']:
                            self.alert_history[person_id]['fallen'] = current_time
                            
                            return {
                                'type': 'fallen',
                                'severity': 'critical',
                                'person_id': int(person_id),
                                'aspect_ratio': float(current_ratio),
                                'y_change': float(y_change),
                                'time_horizontal': float(time_horizontal),
                                'avg_velocity': float(recent_velocity),
                                'position': [int(current_pos[0]), int(current_pos[1])],
                                'confidence': 0.88,
                                'description': f'Person fallen and motionless for {time_horizontal:.1f}s (aspect ratio: {current_ratio:.2f})'
                            }
        else:
            # Reset fall timestamp if person is moving or standing
            track['fall_timestamp'] = None
        
        return None
    
    def detect_fire_smoke_yolo(self, frame, current_time):
        """Detect fire and smoke using YOLO object detection with persistence check"""
        alerts = []
        
        if self.fire_smoke_model is None:
            return self.detect_fire_smoke_color(frame, current_time)
        
        try:
            results = self.fire_smoke_model(frame, conf=0.4, verbose=False)
            
            if len(results[0].boxes) == 0:
                return alerts
            
            boxes = results[0].boxes
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                class_names = {0: 'fire', 1: 'smoke'}
                detection_type = class_names.get(cls)
                
                if detection_type is None:
                    continue
                
                threshold_key = f'{detection_type}_confidence_threshold'
                if conf < self.config[threshold_key]:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                region_id = f'{detection_type}_zone_{center_x//100}_{center_y//100}'
                region_data = self.fire_regions[region_id]
                
                if region_data['first_detected'] is None:
                    region_data['first_detected'] = current_time
                
                region_data['last_detected'] = current_time
                region_data['confidence_history'].append(conf)
                region_data['bbox_history'].append([x1, y1, x2, y2])
                region_data['detection_frames'] += 1
                
                duration = current_time - region_data['first_detected']
                min_frames = self.config['fire_min_frames']
                
                # Require both time persistence AND frame count
                if (duration >= self.config['fire_persistence_threshold'] and 
                    region_data['detection_frames'] >= min_frames and 
                    not region_data['alerted']):
                    
                    region_data['alerted'] = True
                    
                    avg_conf = np.mean(list(region_data['confidence_history']))
                    
                    bbox_area = (x2 - x1) * (y2 - y1)
                    frame_area = frame.shape[0] * frame.shape[1]
                    area_ratio = bbox_area / frame_area
                    
                    severity = 'critical' if detection_type == 'fire' else 'high'
                    if area_ratio > 0.15:
                        severity = 'critical'
                    
                    alerts.append({
                        'type': detection_type,
                        'severity': severity,
                        'zone_id': region_id,
                        'position': [center_x, center_y],
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area_ratio': float(area_ratio * 100),
                        'duration': float(duration),
                        'detection_frames': int(region_data['detection_frames']),
                        'confidence': float(avg_conf),
                        'description': f'{detection_type.capitalize()} detected ({area_ratio*100:.1f}% of frame, {duration:.1f}s) - YOLO',
                        'detection_method': 'yolo_object_detection'
                    })
        
        except Exception as e:
            print(f"[BehaviorDetector] Error in YOLO fire/smoke detection: {e}")
        
        return alerts
    
    def detect_fire_smoke_color(self, frame, current_time):
        """FALLBACK: Color-based fire/smoke detection with persistence"""
        alerts = []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Fire detection (orange/yellow/red hues)
        lower_fire1 = np.array([0, 100, 100])
        upper_fire1 = np.array([20, 255, 255])
        lower_fire2 = np.array([20, 100, 100])
        upper_fire2 = np.array([30, 255, 255])
        
        mask_fire1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask_fire2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        fire_mask = cv2.bitwise_or(mask_fire1, mask_fire2)
        
        # Smoke detection (gray/white diffuse areas)
        lower_smoke = np.array([0, 0, 100])
        upper_smoke = np.array([180, 50, 220])
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        
        frame_area = frame.shape[0] * frame.shape[1]
        fire_area = cv2.countNonZero(fire_mask)
        smoke_area = cv2.countNonZero(smoke_mask)
        
        fire_ratio = fire_area / frame_area
        smoke_ratio = smoke_area / frame_area
        
        # Fire detection with persistence
        if fire_ratio > 0.03:
            zone_id = 'fire_color_zone'
            fire_data = self.fire_regions[zone_id]
            
            if fire_data['first_detected'] is None:
                fire_data['first_detected'] = current_time
            fire_data['last_detected'] = current_time
            fire_data['detection_frames'] += 1
            
            duration = current_time - fire_data['first_detected']
            
            if duration >= 5.0 and fire_data['detection_frames'] >= 30 and not fire_data['alerted']:
                fire_data['alerted'] = True
                
                moments = cv2.moments(fire_mask)
                cx = int(moments['m10'] / moments['m00']) if moments['m00'] > 0 else frame.shape[1] // 2
                cy = int(moments['m01'] / moments['m00']) if moments['m00'] > 0 else frame.shape[0] // 2
                
                alerts.append({
                    'type': 'fire',
                    'severity': 'critical',
                    'zone_id': zone_id,
                    'position': [cx, cy],
                    'area_ratio': float(fire_ratio * 100),
                    'duration': float(duration),
                    'detection_frames': int(fire_data['detection_frames']),
                    'confidence': min(0.75, 0.5 + (fire_ratio * 5)),
                    'description': f'Fire detected ({fire_ratio*100:.1f}% of frame) - Color fallback',
                    'detection_method': 'color_hsv'
                })
        
        # Smoke detection with persistence
        if smoke_ratio > 0.05:
            zone_id = 'smoke_color_zone'
            smoke_data = self.fire_regions[zone_id]
            
            if smoke_data['first_detected'] is None:
                smoke_data['first_detected'] = current_time
            smoke_data['last_detected'] = current_time
            smoke_data['detection_frames'] += 1
            
            duration = current_time - smoke_data['first_detected']
            
            if duration >= 5.0 and smoke_data['detection_frames'] >= 30 and not smoke_data['alerted']:
                smoke_data['alerted'] = True
                
                moments = cv2.moments(smoke_mask)
                cx = int(moments['m10'] / moments['m00']) if moments['m00'] > 0 else frame.shape[1] // 2
                cy = int(moments['m01'] / moments['m00']) if moments['m00'] > 0 else frame.shape[0] // 2
                
                alerts.append({
                    'type': 'smoke',
                    'severity': 'high',
                    'zone_id': zone_id,
                    'position': [cx, cy],
                    'area_ratio': float(smoke_ratio * 100),
                    'duration': float(duration),
                    'detection_frames': int(smoke_data['detection_frames']),
                    'confidence': min(0.70, 0.4 + (smoke_ratio * 3)),
                    'description': f'Smoke detected ({smoke_ratio*100:.1f}% of frame) - Color fallback',
                    'detection_method': 'color_hsv'
                })
        
        return alerts
    
    def detect_crowd_density(self, all_person_bboxes, frame_shape, current_time):
        """
        Detect crowd density with sustained confirmation
        """
        alerts = []
        
        if len(all_person_bboxes) < self.config['crowd_density_threshold']:
            return alerts
        
        centroids = [(bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2) 
                     for bbox in all_person_bboxes]
        
        grid_size = self.config['crowd_area_size']
        h, w = frame_shape[:2]
        
        grid_counts = defaultdict(list)
        
        for centroid in centroids:
            grid_x = int(centroid[0] // grid_size)
            grid_y = int(centroid[1] // grid_size)
            grid_counts[(grid_x, grid_y)].append(centroid)
        
        for zone_key, zone_people in grid_counts.items():
            zone_count = len(zone_people)
            
            zone_id = f"zone_{zone_key[0]}_{zone_key[1]}"
            zone_data = self.crowd_zones[zone_id]
            zone_data['person_count'] = zone_count
            zone_data['density_history'].append(zone_count)
            
            # Cooldown check
            if current_time - zone_data['last_alert'] < self.config['alert_cooldown']:
                continue
            
            # Critical crowd density (immediate alert)
            if zone_count >= self.config['high_density_threshold']:
                zone_data['high_density_frames'] += 1
                
                # Require sustained high density (at least 2 seconds)
                min_frames = int(self.estimated_fps * 2.0)
                
                if zone_data['high_density_frames'] >= min_frames:
                    zone_data['last_alert'] = current_time
                    zone_data['high_density_frames'] = 0  # Reset
                    
                    zone_center = (
                        int(np.mean([p[0] for p in zone_people])),
                        int(np.mean([p[1] for p in zone_people]))
                    )
                    
                    alerts.append({
                        'type': 'crowd',
                        'severity': 'critical',
                        'person_count': int(zone_count),
                        'zone_id': zone_id,
                        'zone_center': [zone_center[0], zone_center[1]],
                        'confidence': 0.95,
                        'description': f'Critical crowd density: {zone_count} people in {grid_size}x{grid_size}px area'
                    })
            
            # Elevated crowd density (requires longer confirmation)
            elif zone_count >= self.config['crowd_density_threshold']:
                min_frames = self.config['crowd_min_frames']
                
                if len(zone_data['density_history']) >= min_frames:
                    avg_density = np.mean(list(zone_data['density_history']))
                    
                    if avg_density >= self.config['crowd_density_threshold']:
                        zone_data['last_alert'] = current_time
                        
                        zone_center = (
                            int(np.mean([p[0] for p in zone_people])),
                            int(np.mean([p[1] for p in zone_people]))
                        )
                        
                        alerts.append({
                            'type': 'crowd',
                            'severity': 'medium',
                            'person_count': int(zone_count),
                            'zone_id': zone_id,
                            'zone_center': [int(zone_center[0]), int(zone_center[1])],
                            'avg_density': float(avg_density),
                            'confidence': 0.82,
                            'description': f'Elevated crowd density: {zone_count} people (avg: {avg_density:.1f})'
                        })
            else:
                # Reset high density counter if below threshold
                zone_data['high_density_frames'] = 0
        
        return alerts
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _extract_keypoints_by_index(self, pose_results, idx):
        """Extract keypoints for a specific detection index"""
        try:
            if pose_results[0].keypoints is None or len(pose_results[0].keypoints.xy) <= idx:
                return None
            
            keypoints = pose_results[0].keypoints.xy.cpu().numpy()[idx]
            
            return {
                'nose': keypoints[0],
                'left_shoulder': keypoints[5],
                'right_shoulder': keypoints[6],
                'left_elbow': keypoints[7],
                'right_elbow': keypoints[8],
                'left_wrist': keypoints[9],
                'right_wrist': keypoints[10],
                'left_hip': keypoints[11],
                'right_hip': keypoints[12]
            }
        except Exception as e:
            return None
    
    def is_person_violent(self, person_id):
        """
        Check if a person is currently marked as violent
        
        Returns:
            bool: True if person is violent (face blur should be removed)
        """
        if person_id not in self.person_tracks:
            return False
        
        track = self.person_tracks[person_id]
        return track.get('is_violent', False)
    
    def get_violent_persons(self):
        """
        Get list of all currently violent person IDs
        
        Returns:
            list: Person IDs marked as violent
        """
        violent_ids = []
        for person_id, track in self.person_tracks.items():
            if track.get('is_violent', False):
                violent_ids.append(person_id)
        return violent_ids
    
    def analyze_frame(self, person_tracks_data, frame, current_time, pose_results=None):
        """
        Main analysis function - analyzes all persons in current frame
        
        Args:
            person_tracks_data: List of dicts with 'id' and 'bbox' keys
            frame: Current video frame for fire/smoke detection
            current_time: Current timestamp
            pose_results: YOLO-Pose results (optional, for violence detection)
            
        Returns: 
            List of detected behaviors/alerts
        """
        self.stats['total_frames_processed'] += 1
        alerts = []
        
        # Update all person tracks
        all_bboxes = []
        all_positions = {}
        person_keypoints = {}
        
        for person_data in person_tracks_data:
            person_id = person_data['id']
            bbox = person_data['bbox']
            all_bboxes.append(bbox)
            
            # Extract keypoints if available
            keypoints = None
            if pose_results and len(pose_results) > 0:
                try:
                    best_match_idx = None
                    best_iou = 0.3  # Minimum IOU threshold
                    
                    person_bbox_xyxy = [
                        bbox[0], bbox[1], 
                        bbox[0] + bbox[2], bbox[1] + bbox[3]
                    ]
                    
                    if hasattr(pose_results[0], 'boxes') and len(pose_results[0].boxes) > 0:
                        for idx, pose_box in enumerate(pose_results[0].boxes):
                            pose_bbox = pose_box.xyxy[0].cpu().numpy()
                            iou = self._calculate_iou(person_bbox_xyxy, pose_bbox)
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_match_idx = idx
                        
                        if best_match_idx is not None:
                            keypoints = self._extract_keypoints_by_index(pose_results, best_match_idx)
                            person_keypoints[person_id] = keypoints
                except Exception as e:
                    pass
            
            self.update_person(person_id, bbox, current_time, keypoints)
            
            centroid = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
            all_positions[person_id] = centroid
        
        # Detect individual behaviors
        for person_data in person_tracks_data:
            person_id = person_data['id']
            
            # Loitering detection
            loitering = self.detect_loitering(person_id, current_time)
            if loitering:
                alerts.append(loitering)
            
            # Running detection
            running = self.detect_running(person_id, current_time)
            if running:
                alerts.append(running)
            
            # Violence detection (POSE-BASED) - DISABLES FACE BLUR
            keypoints = person_keypoints.get(person_id)
            violence = self.detect_violence_pose(person_id, current_time, all_positions, keypoints)
            if violence:
                alerts.append(violence)
            
            # Fallen person detection
            fallen = self.detect_fallen_person(person_id, current_time)
            if fallen:
                alerts.append(fallen)
        
        # Crowd density detection
        crowd_alerts = self.detect_crowd_density(all_bboxes, frame.shape, current_time)
        alerts.extend(crowd_alerts)
        
        # Fire/Smoke detection (YOLO-BASED)
        fire_smoke_alerts = self.detect_fire_smoke_yolo(frame, current_time)
        alerts.extend(fire_smoke_alerts)
        
        # Log statistics periodically
        if self.stats['total_frames_processed'] % 1000 == 0:
            self.print_statistics()
        
        return alerts
    
    def cleanup_old_tracks(self, active_person_ids, max_age=30.0):
        """Remove tracking data for persons not seen recently"""
        current_time = time.time()
        to_remove = []
        
        for person_id, track in self.person_tracks.items():
            if person_id not in active_person_ids:
                if track['last_seen'] and (current_time - track['last_seen']) > max_age:
                    to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_tracks[person_id]
            if person_id in self.alert_history:
                del self.alert_history[person_id]
    
    def reset(self):
        """Reset all tracking data"""
        self.person_tracks.clear()
        self.crowd_zones.clear()
        self.fire_regions.clear()
        self.alert_history.clear()
        self.fps_history.clear()
        
        # Reset statistics
        self.stats = {
            'total_frames_processed': 0,
            'violence_false_positive_filtered': 0,
            'running_false_positive_filtered': 0,
            'velocity_spikes_filtered': 0
        }
        
        # Reset face blur system
        if FACE_BLUR_INTEGRATION:
            try:
                reset_violence_state()
                print("[BehaviorDetector] ✅ Violence state and face blur reset")
            except Exception as e:
                print(f"[BehaviorDetector] ⚠️ Failed to reset violence state: {e}")
    
    def get_statistics(self):
        """Get behavior detection statistics"""
        violent_count = len(self.get_violent_persons())
        
        return {
            'active_tracks': len(self.person_tracks),
            'violent_persons': violent_count,
            'crowd_zones': len(self.crowd_zones),
            'fire_regions': len(self.fire_regions),
            'total_alerts': sum(len(alerts) for alerts in self.alert_history.values()),
            'face_blur_integration': FACE_BLUR_INTEGRATION,
            'estimated_fps': round(self.estimated_fps, 1),
            'total_frames_processed': self.stats['total_frames_processed'],
            'violence_false_positives_filtered': self.stats['violence_false_positive_filtered'],
            'running_false_positives_filtered': self.stats['running_false_positive_filtered'],
            'velocity_spikes_filtered': self.stats['velocity_spikes_filtered']
        }
    
    def print_statistics(self):
        """Print detection statistics"""
        stats = self.get_statistics()
        print(f"\n[BehaviorDetector] 📊 STATISTICS (after {stats['total_frames_processed']} frames)")
        print(f"  FPS: {stats['estimated_fps']}")
        print(f"  Active tracks: {stats['active_tracks']}")
        print(f"  Violent persons: {stats['violent_persons']}")
        print(f"  False positives filtered:")
        print(f"    - Violence: {stats['violence_false_positives_filtered']}")
        print(f"    - Running: {stats['running_false_positives_filtered']}")
        print(f"    - Velocity spikes: {stats['velocity_spikes_filtered']}")
        print(f"  Face blur integration: {'✅ Enabled' if stats['face_blur_integration'] else '❌ Disabled'}\n")


# Alias for backward compatibility
BehaviorDetector = EnhancedBehaviorDetector
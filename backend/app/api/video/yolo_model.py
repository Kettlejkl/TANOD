# app/api/video/yolo_model.py
# COMPLETE VERSION: Face blur with violence detection integration

from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os
import time
from collections import defaultdict, deque

# ============================================================================
# CONFIGURATION
# ============================================================================

DETECTION_MODE = "ULTRA_FAST"
YOLO_WEIGHTS = "yolov8n-pose.pt"
CONF_THRESHOLD = 0.65

# ============================================================================
# YOLO MODEL INITIALIZATION
# ============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[YOLO] Using device: {device}")

try:
    model = YOLO(YOLO_WEIGHTS)
    model.to(device)
    print(f"[YOLO] Model loaded: {YOLO_WEIGHTS}")
except Exception as e:
    print(f"[YOLO] Error loading model: {e}")
    model = None

# ============================================================================
# FACE DETECTION SETUP
# ============================================================================

FACE_DETECTOR_AVAILABLE = False
face_cascade = None

try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if not face_cascade.empty():
        FACE_DETECTOR_AVAILABLE = True
        print("[FaceBlur] ✅ Face detector loaded successfully")
    else:
        print("[FaceBlur] ⚠️ Face cascade file is empty")
except Exception as e:
    print(f"[FaceBlur] ⚠️ Could not load face detector: {e}")


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    return intersection / union


# ============================================================================
# FACE TRACKER
# ============================================================================

class SmoothFaceTracker:
    """
    Tracks faces across frames to reduce jitter
    """
    def __init__(self, max_disappeared=15, iou_threshold=0.15):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
    
    def register(self, bbox):
        """Register a new face"""
        self.objects[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove a face that's been gone too long"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections):
        """
        Update tracked faces with new detections
        
        Args:
            detections: List of (x, y, w, h) face bounding boxes
            
        Returns:
            List of tracked face bounding boxes
        """
        # No detections - increment disappeared counters
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return list(self.objects.values())
        
        # Convert detections to xyxy format for IOU calculation
        detection_boxes = []
        for (x, y, w, h) in detections:
            detection_boxes.append([x, y, x + w, y + h])
        
        # No existing objects - register all detections
        if len(self.objects) == 0:
            for bbox in detections:
                self.register(bbox)
        else:
            # Match existing objects with new detections
            object_ids = list(self.objects.keys())
            object_boxes = []
            
            for object_id in object_ids:
                x, y, w, h = self.objects[object_id]
                object_boxes.append([x, y, x + w, y + h])
            
            # Calculate IOU matrix
            matched_detections = set()
            matched_objects = set()
            
            for obj_idx, obj_box in enumerate(object_boxes):
                best_iou = self.iou_threshold
                best_det_idx = -1
                
                for det_idx, det_box in enumerate(detection_boxes):
                    if det_idx in matched_detections:
                        continue
                    
                    iou = calculate_iou(obj_box, det_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = det_idx
                
                if best_det_idx >= 0:
                    # Update existing object
                    object_id = object_ids[obj_idx]
                    self.objects[object_id] = detections[best_det_idx]
                    self.disappeared[object_id] = 0
                    matched_detections.add(best_det_idx)
                    matched_objects.add(obj_idx)
            
            # Handle unmatched objects
            for obj_idx, object_id in enumerate(object_ids):
                if obj_idx not in matched_objects:
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Register new detections
            for det_idx, bbox in enumerate(detections):
                if det_idx not in matched_detections:
                    self.register(bbox)
        
        return list(self.objects.values())


# ============================================================================
# VIOLENCE STATE TRACKER
# ============================================================================

class ViolenceStateTracker:
    """
    Tracks violence detection state to control face blur
    """
    def __init__(self, violence_memory_duration=30.0):
        """
        Args:
            violence_memory_duration: How long (seconds) to keep blur disabled after violence
        """
        self.violence_memory_duration = violence_memory_duration
        self.active_violence_zones = {}  # {zone_id: timestamp}
        self.violence_person_ids = set()  # Person IDs involved in violence
        self.last_violence_time = 0
    
    def register_violence(self, person_ids, zone_id=None):
        """
        Register a violence event
        
        Args:
            person_ids: List of person IDs involved
            zone_id: Optional zone identifier
        """
        current_time = time.time()
        self.last_violence_time = current_time
        
        # Track persons involved
        for pid in person_ids:
            self.violence_person_ids.add(pid)
        
        # Track zone
        if zone_id:
            self.active_violence_zones[zone_id] = current_time
    
    def cleanup_expired_violence(self):
        """Remove old violence records"""
        current_time = time.time()
        
        # Clean up old zones
        expired_zones = [
            zone_id for zone_id, timestamp in self.active_violence_zones.items()
            if current_time - timestamp > self.violence_memory_duration
        ]
        for zone_id in expired_zones:
            del self.active_violence_zones[zone_id]
        
        # Clean up person IDs if global violence has expired
        if current_time - self.last_violence_time > self.violence_memory_duration:
            self.violence_person_ids.clear()
    
    def should_blur_faces(self):
        """
        Check if faces should be blurred
        
        Returns:
            bool: True if blur should be applied, False if violence detected recently
        """
        self.cleanup_expired_violence()
        
        # Disable blur if any violence detected recently
        current_time = time.time()
        if current_time - self.last_violence_time < self.violence_memory_duration:
            return False
        
        return True
    
    def is_person_in_violence(self, person_id):
        """Check if specific person is involved in violence"""
        return person_id in self.violence_person_ids
    
    def get_violence_status(self):
        """Get current violence status for display"""
        self.cleanup_expired_violence()
        return {
            'blur_enabled': self.should_blur_faces(),
            'active_zones': len(self.active_violence_zones),
            'involved_persons': len(self.violence_person_ids),
            'last_violence_ago': time.time() - self.last_violence_time if self.last_violence_time > 0 else None
        }
    
    def reset(self):
        """Reset all violence state"""
        self.active_violence_zones.clear()
        self.violence_person_ids.clear()
        self.last_violence_time = 0


# Global violence state tracker
_violence_tracker = ViolenceStateTracker(violence_memory_duration=30.0)


# ============================================================================
# OPTIMIZED FACE BLURRER
# ============================================================================

class OptimizedFaceBlurrer:
    """
    Configurable face detection with violence-aware blur control
    """
    def __init__(self, blur_every_n_frames=2, detection_mode="BALANCED", 
                 respect_violence_state=True):
        self.blur_every_n_frames = blur_every_n_frames
        self.frame_count = 0
        self.detection_mode = detection_mode
        self.respect_violence_state = respect_violence_state
        
        self.face_tracker = SmoothFaceTracker(
            max_disappeared=15,
            iou_threshold=0.15
        )
        
        self.blur_kernels = {
            'tiny': (21, 21),
            'small': (31, 31),
            'medium': (51, 51),
            'large': (71, 71),
            'xlarge': (91, 91),
        }
        self.sigma = 30
        
        self._configure_detection_mode()
    
    def _configure_detection_mode(self):
        """Configure detection parameters based on performance mode"""
        if self.detection_mode == "MAXIMUM":
            self.detection_params = [
                (1.03, 2, (15, 15)),
                (1.05, 2, (20, 20)),
                (1.08, 2, (25, 25)),
                (1.1, 3, (30, 30)),
                (1.12, 3, (35, 35)),
                (1.15, 3, (40, 40)),
                (1.18, 4, (45, 45)),
                (1.2, 4, (50, 50)),
                (1.25, 4, (60, 60)),
                (1.3, 5, (70, 70)),
            ]
            self.use_histogram_eq = True
            self.nms_threshold = 0.25
        
        elif self.detection_mode == "BALANCED":
            self.detection_params = [
                (1.05, 2, (20, 20)),
                (1.08, 2, (25, 25)),
                (1.1, 3, (30, 30)),
                (1.15, 3, (40, 40)),
                (1.2, 4, (50, 50)),
                (1.25, 4, (60, 60)),
            ]
            self.use_histogram_eq = True
            self.nms_threshold = 0.3
        
        elif self.detection_mode == "FAST":
            self.detection_params = [
                (1.08, 2, (25, 25)),
                (1.1, 3, (30, 30)),
                (1.15, 3, (40, 40)),
            ]
            self.use_histogram_eq = True
            self.nms_threshold = 0.35
        
        elif self.detection_mode == "ULTRA_FAST":
            self.detection_params = [
                (1.1, 3, (30, 30)),
                (1.2, 4, (50, 50)),
            ]
            self.use_histogram_eq = False
            self.nms_threshold = 0.4
        
        else:
            self.detection_mode = "BALANCED"
            self._configure_detection_mode()
    
    def _get_blur_kernel(self, face_size):
        if face_size < 60:
            return self.blur_kernels['tiny']
        elif face_size < 100:
            return self.blur_kernels['small']
        elif face_size < 180:
            return self.blur_kernels['medium']
        elif face_size < 300:
            return self.blur_kernels['large']
        else:
            return self.blur_kernels['xlarge']
    
    def _detect_faces(self, gray):
        """Detect faces using configured mode"""
        if not FACE_DETECTOR_AVAILABLE or face_cascade is None:
            return []
        
        all_faces = []
        
        if self.use_histogram_eq:
            gray_enhanced = cv2.equalizeHist(gray)
            gray_variants = [gray, gray_enhanced]
        else:
            gray_variants = [gray]
        
        for gray_variant in gray_variants:
            for scale_factor, min_neighbors, min_size in self.detection_params:
                faces = face_cascade.detectMultiScale(
                    gray_variant,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) > 0:
                    all_faces.extend(faces)
        
        if len(all_faces) > 0:
            all_faces = self._apply_nms(all_faces, threshold=self.nms_threshold)
        
        return all_faces
    
    def _apply_nms(self, faces, threshold=0.3):
        """Apply Non-Maximum Suppression"""
        if len(faces) == 0:
            return []
        
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h, w * h])
        
        boxes = np.array(boxes)
        indices = np.argsort(boxes[:, 4])[::-1]
        
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            ious = []
            for j in indices[1:]:
                iou = calculate_iou(boxes[i, :4], boxes[j, :4])
                ious.append(iou)
            
            ious = np.array(ious)
            indices = indices[1:][ious < threshold]
        
        result = []
        for i in keep:
            x1, y1, x2, y2, _ = boxes[i]
            result.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        
        return result
    
    def blur_faces(self, frame, violence_detected=None):
        """
        Blur faces with violence-aware control
        
        Args:
            frame: Input frame
            violence_detected: Optional - force blur state (True=blur, False=no blur)
        
        Returns:
            Frame with faces blurred (if appropriate)
        """
        if not FACE_DETECTOR_AVAILABLE or face_cascade is None:
            return frame
        
        # Check violence state
        if self.respect_violence_state:
            if violence_detected is None:
                # Use global violence tracker
                should_blur = _violence_tracker.should_blur_faces()
            else:
                # Use provided state
                should_blur = violence_detected
            
            if not should_blur:
                # Violence detected - skip blur and show clear faces
                return frame
        
        self.frame_count += 1
        
        try:
            if self.frame_count % self.blur_every_n_frames == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = self._detect_faces(gray)
            else:
                detected_faces = []
            
            tracked_faces = self.face_tracker.update(detected_faces)
            
            for (x, y, w, h) in tracked_faces:
                base_padding = 0.35
                distance_factor = (w / frame.shape[1]) * 0.5
                padding_ratio = min(0.6, base_padding + distance_factor)
                
                padding_x = int(w * padding_ratio)
                padding_y_top = int(h * (padding_ratio + 0.25))
                padding_y_bottom = int(h * (padding_ratio + 0.15))
                
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y_top)
                x2 = min(frame.shape[1], x + w + padding_x)
                y2 = min(frame.shape[0], y + h + padding_y_bottom)
                
                face_region = frame[y1:y2, x1:x2]
                
                if face_region.size > 0:
                    face_size = max(w, h)
                    kernel = self._get_blur_kernel(face_size)
                    adaptive_sigma = min(50, self.sigma + (face_size / 20))
                    
                    blurred = cv2.GaussianBlur(
                        face_region,
                        kernel,
                        adaptive_sigma
                    )
                    
                    frame[y1:y2, x1:x2] = blurred
            
            return frame
            
        except Exception as e:
            print(f"[FaceBlur] Error blurring faces: {e}")
            return frame
    
    def reset_tracker(self):
        self.face_tracker = SmoothFaceTracker(
            max_disappeared=15,
            iou_threshold=0.15
        )


# Create global instance with violence awareness
_face_blurrer = OptimizedFaceBlurrer(
    blur_every_n_frames=1,
    detection_mode=DETECTION_MODE,
    respect_violence_state=True
)


# ============================================================================
# PUBLIC API FUNCTIONS
# ============================================================================

def blur_faces(frame, violence_detected=None):
    """
    Main function to blur faces (violence-aware)
    
    Args:
        frame: Input frame
        violence_detected: Optional - True to skip blur, False to force blur, None to check global state
    
    Returns:
        Frame with conditional face blur
    """
    return _face_blurrer.blur_faces(frame, violence_detected)


def register_violence_event(person_ids, zone_id=None):
    """
    Register a violence event (disables face blur temporarily)
    
    Args:
        person_ids: List of person IDs involved in violence
        zone_id: Optional zone identifier
    """
    _violence_tracker.register_violence(person_ids, zone_id)
    print(f"[FaceBlur] ⚠️ Violence detected - Face blur DISABLED for 30s")


def reset_violence_state():
    """Reset violence state (re-enables face blur)"""
    _violence_tracker.reset()
    print(f"[FaceBlur] ✅ Violence state reset - Face blur ENABLED")


def get_violence_status():
    """Get current violence/blur status"""
    return _violence_tracker.get_violence_status()


def set_violence_memory_duration(seconds):
    """Set how long to keep blur disabled after violence"""
    _violence_tracker.violence_memory_duration = seconds
    print(f"[FaceBlur] Violence memory set to {seconds}s")


def reset_face_tracker():
    """Reset the face tracker"""
    _face_blurrer.reset_tracker()


# ============================================================================
# ALTERNATIVE BLUR METHODS
# ============================================================================

def blur_faces_pixelate(frame, violence_detected=None):
    """Pixelate faces (violence-aware)"""
    if not FACE_DETECTOR_AVAILABLE or face_cascade is None:
        return frame
    
    # Check violence state
    if violence_detected is None:
        should_blur = _violence_tracker.should_blur_faces()
    else:
        should_blur = violence_detected
    
    if not should_blur:
        return frame
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        for (x, y, w, h) in faces:
            padding = int(w * 0.35)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size > 0:
                small = cv2.resize(face_region, (10, 10), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                frame[y1:y2, x1:x2] = pixelated
        
        return frame
        
    except Exception as e:
        print(f"[FaceBlur] Error pixelating faces: {e}")
        return frame


def blur_faces_black_bar(frame, violence_detected=None):
    """Black bars over faces (violence-aware, fastest method)"""
    if not FACE_DETECTOR_AVAILABLE or face_cascade is None:
        return frame
    
    # Check violence state
    if violence_detected is None:
        should_blur = _violence_tracker.should_blur_faces()
    else:
        should_blur = violence_detected
    
    if not should_blur:
        return frame
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        for (x, y, w, h) in faces:
            padding = int(w * 0.35)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        return frame
        
    except Exception as e:
        print(f"[FaceBlur] Error adding black bars: {e}")
        return frame


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'model',
    'CONF_THRESHOLD',
    'device',
    'blur_faces',
    'register_violence_event',
    'reset_violence_state',
    'get_violence_status',
    'set_violence_memory_duration',
    'reset_face_tracker',
    'blur_faces_pixelate',
    'blur_faces_black_bar',
    'FACE_DETECTOR_AVAILABLE',
    'OptimizedFaceBlurrer',
    'SmoothFaceTracker',
    'ViolenceStateTracker'
]
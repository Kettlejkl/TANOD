from ultralytics import YOLO
import numpy as np
import torch
import cv2
import time
from functools import lru_cache
from threading import Thread
from queue import Queue

DETECTION_MODE = "HYBRID"
YOLO_WEIGHTS = "yolov8n-pose.pt"
CONF_THRESHOLD = 0.50

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[YOLO] Using device: {device}")

# Face detection model config
FACE_CONF_THRESHOLD = 0.50
FACE_IOU_THRESHOLD  = 0.45
FACE_IMG_SIZE       = 320

# Dedicated face detection model (optional — set to None to use pose-only)
face_model = None
try:
    face_model = YOLO("yolov8n-face.pt")
    face_model.to(device)
    print("[FaceBlur] ✅ Face model loaded: yolov8n-face.pt")
except Exception as e:
    print(f"[FaceBlur] ⚠️ No face model loaded ({e}), using pose keypoints only")

# AGGRESSIVE GPU Optimization Settings
if device == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)
    torch.cuda.empty_cache()
    print(f"[YOLO] ⚡ AGGRESSIVE GPU Optimizations enabled")
    print(f"[YOLO] GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

try:
    model = YOLO(YOLO_WEIGHTS)
    model.to(device)

    if device == 'cuda':
        use_half = False
        try:
            model.model.half()
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model(dummy, conf=CONF_THRESHOLD, verbose=False, imgsz=640, half=True)
            use_half = True
            print(f"[YOLO] ✅ FP16 enabled")
        except:
            model.model.float()
            print(f"[YOLO] ⚠️ Using FP32")

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        _ = model(dummy, conf=CONF_THRESHOLD, verbose=False, imgsz=640, half=use_half)

    print(f"[YOLO] ✅ Model warmed up (5 iterations)")
    print(f"[YOLO] Model loaded: {YOLO_WEIGHTS}")
except Exception as e:
    print(f"[YOLO] Error loading model: {e}")
    model = None
    use_half = False

FACE_DETECTOR_AVAILABLE = False
face_cascade = None

try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if not face_cascade.empty():
        FACE_DETECTOR_AVAILABLE = True
        print("[FaceBlur] ✅ Haar Cascade loaded")
    else:
        print("[FaceBlur] ⚠️ Face cascade file is empty")
except Exception as e:
    print(f"[FaceBlur] ⚠️ Could not load face detector: {e}")


@lru_cache(maxsize=128)
def get_blur_kernel(face_size):
    if face_size < 80:
        return (21, 21)
    elif face_size < 150:
        return (31, 31)
    else:
        return (45, 45)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def calculate_distance(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    center1_x = x1 + w1 / 2
    center1_y = y1 + h1 / 2
    center2_x = x2 + w2 / 2
    center2_y = y2 + h2 / 2
    return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)


def extract_face_from_pose(keypoints):
    try:
        facial_keypoints = keypoints[:3]
        valid_kpts = [(x, y) for x, y in facial_keypoints if x > 0 and y > 0]
        if len(valid_kpts) < 2:
            return None

        x_coords = [kpt[0] for kpt in valid_kpts]
        y_coords = [kpt[1] for kpt in valid_kpts]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        width = max_x - min_x
        height = max_y - min_y

        # Tighter expansion — just enough to cover the face
        expand_x = max(width * 0.5, 20)
        expand_y_top = max(height * 0.3, 15)
        expand_y_bottom = max(height * 0.6, 25)

        x = int(min_x - expand_x / 2)
        y = int(min_y - expand_y_top)
        w = int(width + expand_x)
        h = int(height + expand_y_top + expand_y_bottom)

        if w < 20 or h < 20 or w > 400 or h > 400:
            return None
        return (x, y, w, h)
    except Exception:
        return None


def merge_face_detections(haar_faces, pose_faces, frame_shape):
    if not pose_faces:
        return haar_faces
    if not haar_faces:
        return pose_faces
    merged = list(pose_faces)
    for haar_face in haar_faces:
        hx, hy, hw, hh = haar_face
        haar_box = [hx, hy, hx + hw, hy + hh]
        overlaps = False
        for pose_face in pose_faces:
            px, py, pw, ph = pose_face
            pose_box = [px, py, px + pw, py + ph]
            if calculate_iou(haar_box, pose_box) > 0.15:
                overlaps = True
                break
        if not overlaps:
            merged.append(haar_face)
    return merged


class AsyncYOLOInference:
    """Asynchronous YOLO inference — runs on a dedicated thread so the main
    display loop is never blocked waiting for GPU results."""

    def __init__(self, model, conf_threshold, use_half):
        self.model = model
        self.conf_threshold = conf_threshold
        self.use_half = use_half
        self.input_queue = Queue(maxsize=2)   # small buffer — drop stale frames
        self.output_queue = Queue(maxsize=2)
        self.running = True
        self.thread = Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        print("[AsyncYOLO] ✅ Async inference thread started")

    def _inference_loop(self):
        while self.running:
            try:
                camera_id, frame = self.input_queue.get(timeout=0.1)
                with torch.cuda.stream(torch.cuda.Stream()):
                    results = self.model(
                        frame,
                        conf=self.conf_threshold,
                        verbose=False,
                        imgsz=320,          # ← reduced from 640 → big FPS gain
                        half=self.use_half,
                        device=device
                    )
                self.output_queue.put((camera_id, results))
            except Exception:
                pass

    def submit(self, camera_id, frame):
        try:
            # Drop oldest if full so we always process the freshest frame
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()
                except Exception:
                    pass
            self.input_queue.put_nowait((camera_id, frame))
            return True
        except Exception:
            return False

    def get_result(self, timeout=0.001):
        try:
            return self.output_queue.get(timeout=timeout)
        except Exception:
            return None

    def stop(self):
        self.running = False
        self.thread.join()


_async_inference = None
if model is not None:
    _async_inference = AsyncYOLOInference(model, CONF_THRESHOLD, use_half)


class SmoothFaceTracker:
    """Tracks faces across frames so detection can run at a lower rate
    while blurring still happens every frame using interpolated positions."""

    def __init__(self, max_disappeared=15, iou_threshold=0.15, distance_threshold=150):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.velocities = {}
        self.last_positions = {}
        self.expansion_factors = {}

    def register(self, bbox):
        self.objects[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.velocities[self.next_object_id] = (0, 0)
        self.last_positions[self.next_object_id] = bbox
        self.expansion_factors[self.next_object_id] = 0.0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        self.velocities.pop(object_id, None)
        self.last_positions.pop(object_id, None)
        self.expansion_factors.pop(object_id, None)

    def predict_position(self, object_id):
        x, y, w, h = self.objects[object_id]
        vx, vy = self.velocities.get(object_id, (0, 0))
        return (int(x + vx), int(y + vy), w, h)

    def update_velocity(self, object_id, new_bbox):
        if object_id not in self.last_positions:
            self.velocities[object_id] = (0, 0)
            self.last_positions[object_id] = new_bbox
            return
        old_x, old_y, _, _ = self.last_positions[object_id]
        new_x, new_y, _, _ = new_bbox
        vx = new_x - old_x
        vy = new_y - old_y
        old_vx, old_vy = self.velocities.get(object_id, (0, 0))
        alpha = 0.7
        self.velocities[object_id] = (
            alpha * vx + (1 - alpha) * old_vx,
            alpha * vy + (1 - alpha) * old_vy,
        )
        self.last_positions[object_id] = new_bbox
        speed = np.sqrt(vx**2 + vy**2)
        # Cap expansion so blur circle doesn't grow too large when moving fast
        self.expansion_factors[object_id] = min(0.1, speed / 300.0)

    def get_expanded_bbox(self, bbox, object_id):
        x, y, w, h = bbox
        expansion = self.expansion_factors.get(object_id, 0.0)
        expand_w = int(w * expansion)
        expand_h = int(h * expansion)
        return (x - expand_w // 2, y - expand_h // 2, w + expand_w, h + expand_h)

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            # Return predicted positions so blur persists smoothly
            return [self.predict_position(oid) for oid in self.objects]

        detection_boxes = [[x, y, x + w, y + h] for (x, y, w, h) in detections]

        if len(self.objects) == 0:
            for bbox in detections:
                self.register(bbox)
        else:
            object_ids = list(self.objects.keys())
            predicted_boxes = []
            for obj_id in object_ids:
                pred_x, pred_y, pred_w, pred_h = self.predict_position(obj_id)
                predicted_boxes.append([pred_x, pred_y, pred_x + pred_w, pred_y + pred_h])

            matched_detections = set()
            matched_objects = set()

            for obj_idx, (obj_id, pred_box) in enumerate(zip(object_ids, predicted_boxes)):
                best_iou = self.iou_threshold
                best_det_idx = -1
                for det_idx, det_box in enumerate(detection_boxes):
                    if det_idx in matched_detections:
                        continue
                    iou = calculate_iou(pred_box, det_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = det_idx
                if best_det_idx >= 0:
                    self.objects[obj_id] = detections[best_det_idx]
                    self.update_velocity(obj_id, detections[best_det_idx])
                    self.disappeared[obj_id] = 0
                    matched_detections.add(best_det_idx)
                    matched_objects.add(obj_idx)

            for obj_idx, obj_id in enumerate(object_ids):
                if obj_idx in matched_objects:
                    continue
                pred_bbox = self.predict_position(obj_id)
                best_distance = self.distance_threshold
                best_det_idx = -1
                for det_idx, det_bbox in enumerate(detections):
                    if det_idx in matched_detections:
                        continue
                    dist = calculate_distance(pred_bbox, det_bbox)
                    if dist < best_distance:
                        best_distance = dist
                        best_det_idx = det_idx
                if best_det_idx >= 0:
                    self.objects[obj_id] = detections[best_det_idx]
                    self.update_velocity(obj_id, detections[best_det_idx])
                    self.disappeared[obj_id] = 0
                    matched_detections.add(best_det_idx)
                    matched_objects.add(obj_idx)

            for obj_idx, obj_id in enumerate(object_ids):
                if obj_idx not in matched_objects:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)

            for det_idx, bbox in enumerate(detections):
                if det_idx not in matched_detections:
                    self.register(bbox)

        expanded = []
        for obj_id, bbox in self.objects.items():
            expanded.append(self.get_expanded_bbox(bbox, obj_id))
        return expanded


class PerCameraViolenceTracker:
    def __init__(self, violence_memory_duration=30.0):
        self.violence_memory_duration = violence_memory_duration
        self.camera_violence_states = {}

    def register_violence(self, camera_id, person_ids, zone_id=None):
        current_time = time.time()
        if camera_id not in self.camera_violence_states:
            self.camera_violence_states[camera_id] = {
                'last_violence_time': 0,
                'violence_person_ids': set(),
                'active_zones': {}
            }
        state = self.camera_violence_states[camera_id]
        state['last_violence_time'] = current_time
        for pid in person_ids:
            state['violence_person_ids'].add(pid)
        if zone_id:
            state['active_zones'][zone_id] = current_time
        print(f"[FaceBlur] ⚠️ Violence detected on {camera_id} - Face blur DISABLED for 30s")

    def cleanup_expired_violence(self, camera_id):
        if camera_id not in self.camera_violence_states:
            return
        current_time = time.time()
        state = self.camera_violence_states[camera_id]
        expired_zones = [
            z for z, t in state['active_zones'].items()
            if current_time - t > self.violence_memory_duration
        ]
        for z in expired_zones:
            del state['active_zones'][z]
        if current_time - state['last_violence_time'] > self.violence_memory_duration:
            state['violence_person_ids'].clear()

    def should_blur_faces(self, camera_id):
        if camera_id not in self.camera_violence_states:
            return True
        self.cleanup_expired_violence(camera_id)
        state = self.camera_violence_states[camera_id]
        return time.time() - state['last_violence_time'] >= self.violence_memory_duration

    def is_person_in_violence(self, camera_id, person_id):
        if camera_id not in self.camera_violence_states:
            return False
        return person_id in self.camera_violence_states[camera_id]['violence_person_ids']

    def get_violence_status(self, camera_id=None):
        if camera_id is None:
            return {
                cam_id: {
                    'blur_enabled': self.should_blur_faces(cam_id),
                    'active_zones': len(state['active_zones']),
                    'involved_persons': len(state['violence_person_ids']),
                    'last_violence_ago': time.time() - state['last_violence_time'] if state['last_violence_time'] > 0 else None
                }
                for cam_id, state in self.camera_violence_states.items()
            }
        if camera_id not in self.camera_violence_states:
            return {'blur_enabled': True, 'active_zones': 0, 'involved_persons': 0, 'last_violence_ago': None}
        self.cleanup_expired_violence(camera_id)
        state = self.camera_violence_states[camera_id]
        return {
            'blur_enabled': self.should_blur_faces(camera_id),
            'active_zones': len(state['active_zones']),
            'involved_persons': len(state['violence_person_ids']),
            'last_violence_ago': time.time() - state['last_violence_time'] if state['last_violence_time'] > 0 else None
        }

    def reset(self, camera_id=None):
        if camera_id is None:
            self.camera_violence_states.clear()
            print(f"[FaceBlur] ✅ Violence state reset for ALL cameras - Face blur ENABLED")
        else:
            self.camera_violence_states.pop(camera_id, None)
            print(f"[FaceBlur] ✅ Violence state reset for {camera_id} - Face blur ENABLED")


_violence_tracker = PerCameraViolenceTracker(violence_memory_duration=30.0)


class HybridFaceBlurrer:
    """
    High-FPS face blurrer with circular blur.

    Key optimisations vs the original:
    - Detection runs every N frames (default 3), blurring runs every frame
      using tracked/predicted positions → smooth video at full frame rate.
    - YOLO imgsz reduced to 320 in the async thread.
    - Haar detection runs on a half-resolution grey image.
    - Circular mask is applied via a fast ellipse + bitwise approach instead
      of per-pixel numpy math.
    - Smaller Gaussian kernels (still looks good, much faster).
    - Blur kernel sigma tuned down.
    """

    # How often (in frames) to run the heavy detectors
    DETECT_EVERY_N = 3

    def __init__(self, blur_every_n_frames=1, detection_mode="HYBRID", respect_violence_state=True):
        self.blur_every_n_frames = blur_every_n_frames
        self.detection_mode = detection_mode
        self.respect_violence_state = respect_violence_state

        self.camera_trackers = {}
        self.camera_frame_counts = {}

        self.sigma = 8
        self._configure_detection_mode()

        self.yolo_results_cache = {}
        self._mask_cache = {}
        self._blur_kernel_cache = {21: (21, 21), 31: (31, 31), 45: (45, 45)}

        print(f"[FaceBlur] 🔀 HYBRID mode — HIGH FPS circular blur (detect every {self.DETECT_EVERY_N} frames)")

    def _get_or_create_tracker(self, camera_id):
        if camera_id not in self.camera_trackers:
            self.camera_trackers[camera_id] = SmoothFaceTracker(
                max_disappeared=15,
                iou_threshold=0.15,
                distance_threshold=150
            )
            self.camera_frame_counts[camera_id] = 0
        return self.camera_trackers[camera_id]

    def _configure_detection_mode(self):
        self.scale_factor = 1.2
        self.min_neighbors = 3
        self.min_size = (30, 30)

    def _get_blur_kernel(self, face_size):
        if face_size < 80:
            return self._blur_kernel_cache[21]
        elif face_size < 150:
            return self._blur_kernel_cache[31]
        else:
            return self._blur_kernel_cache[45]

    def _detect_faces_haar(self, gray_half):
        """Run Haar on a half-resolution image — 4× fewer pixels."""
        if not FACE_DETECTOR_AVAILABLE or face_cascade is None:
            return []
        try:
            faces = face_cascade.detectMultiScale(
                gray_half,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # Scale coords back to full resolution
            return [(x * 2, y * 2, w * 2, h * 2)
                    for (x, y, w, h) in faces if w >= 15 and h >= 15]
        except Exception:
            return []

    def _detect_faces_pose(self, camera_id):
        yolo_results = self.yolo_results_cache.get(camera_id)
        if yolo_results is None:
            return []
        pose_faces = []
        try:
            for result in yolo_results:
                if result.keypoints is None:
                    continue
                keypoints_data = result.keypoints.xy.cpu().numpy()
                for person_kpts in keypoints_data:
                    face_bbox = extract_face_from_pose(person_kpts)
                    if face_bbox is not None:
                        pose_faces.append(face_bbox)
        except Exception:
            pass
        return pose_faces

    def _apply_circular_blur(self, frame, x1, y1, x2, y2, kernel):
        """
        Blur the face region and blend it back using an elliptical mask.
        Uses cv2.ellipse + bitwise ops — much faster than numpy ogrid math.
        """
        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            return

        rh, rw = face_region.shape[:2]

        # Build elliptical mask (cached by size)
        cache_key = (rh, rw)
        mask = self._mask_cache.get(cache_key)
        if mask is None:
            mask = np.zeros((rh, rw), dtype=np.uint8)
            cx, cy = rw // 2, rh // 2
            cv2.ellipse(mask, (cx, cy), (rw // 2, rh // 2), 0, 0, 360, 255, -1)
            # Slight feather via blur on the mask itself
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            if len(self._mask_cache) < 200:
                self._mask_cache[cache_key] = mask

        blurred = cv2.GaussianBlur(face_region, kernel, self.sigma)

        # Alpha-blend using the mask
        alpha = mask.astype(np.float32) / 255.0
        alpha3 = alpha[:, :, np.newaxis]
        frame[y1:y2, x1:x2] = (
            blurred.astype(np.float32) * alpha3 +
            face_region.astype(np.float32) * (1.0 - alpha3)
        ).astype(np.uint8)

    def blur_faces(self, frame, camera_id="default", violence_detected=None, debug=False):
        if self.respect_violence_state:
            should_blur = (
                _violence_tracker.should_blur_faces(camera_id)
                if violence_detected is None else violence_detected
            )
            if not should_blur:
                if debug:
                    cv2.putText(frame, f"Blur: OFF (Violence on {camera_id})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame

        tracker = self._get_or_create_tracker(camera_id)
        self.camera_frame_counts[camera_id] += 1
        frame_count = self.camera_frame_counts[camera_id]

        try:
            # ── Detection phase (throttled) ──────────────────────────────────
            if frame_count % self.DETECT_EVERY_N == 0:
                # Haar on half-res grey
                gray_half = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    (frame.shape[1] // 2, frame.shape[0] // 2)
                )
                haar_faces = self._detect_faces_haar(gray_half)

                # Submit full frame to async YOLO (non-blocking)
                if _async_inference is not None:
                    _async_inference.submit(camera_id, frame)

            # Always drain the async result queue
            if _async_inference is not None:
                result = _async_inference.get_result()
                if result is not None:
                    cam_id, yolo_results = result
                    self.yolo_results_cache[cam_id] = yolo_results

            # Merge detections only on detection frames
            if frame_count % self.DETECT_EVERY_N == 0:
                pose_faces = self._detect_faces_pose(camera_id)
                detected_faces = merge_face_detections(haar_faces, pose_faces, frame.shape)
                tracked_faces = tracker.update(detected_faces)
            else:
                # Non-detection frame: let tracker predict forward
                tracked_faces = tracker.update([])

            if debug:
                cv2.putText(frame, f"{camera_id} | tracked:{len(tracked_faces)} f:{frame_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ── Blur phase (every frame, circular) ───────────────────────────
            for (x, y, w, h) in tracked_faces:
                # Minimal padding — just enough to cover hairline
                pad_x = int(w * 0.05)
                pad_y = int(h * 0.05)

                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)

                kernel = self._get_blur_kernel(max(w, h))
                self._apply_circular_blur(frame, x1, y1, x2, y2, kernel)

            return frame

        except Exception as e:
            print(f"[FaceBlur] Error on {camera_id}: {e}")
            return frame

    def reset_tracker(self, camera_id=None):
        if camera_id is None:
            self.camera_trackers.clear()
            self.camera_frame_counts.clear()
        else:
            self.camera_trackers.pop(camera_id, None)
            self.camera_frame_counts.pop(camera_id, None)


_face_blurrer = HybridFaceBlurrer(
    blur_every_n_frames=1,
    detection_mode="HYBRID",
    respect_violence_state=True
)


# ── Public API ────────────────────────────────────────────────────────────────

def blur_faces(frame, camera_id="default", violence_detected=None, debug=False):
    return _face_blurrer.blur_faces(frame, camera_id, violence_detected, debug)


def force_blur_faces(frame, camera_id="default", debug=False):
    return _face_blurrer.blur_faces(frame, camera_id, violence_detected=False, debug=debug)


def register_violence_event(camera_id, person_ids, zone_id=None):
    _violence_tracker.register_violence(camera_id, person_ids, zone_id)


def reset_violence_state(camera_id=None):
    _violence_tracker.reset(camera_id)


def get_violence_status(camera_id=None):
    return _violence_tracker.get_violence_status(camera_id)


def set_violence_memory_duration(seconds):
    _violence_tracker.violence_memory_duration = seconds


def reset_face_tracker(camera_id=None):
    _face_blurrer.reset_tracker(camera_id)


def set_detection_mode(mode):
    pass  # Fixed to optimised hybrid mode


def blur_faces_pixelate(frame, violence_detected=None):
    if not FACE_DETECTOR_AVAILABLE or face_cascade is None:
        return frame
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=3, minSize=(40, 40))
        for (x, y, w, h) in faces:
            padding = int(w * 0.05)
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
        print(f"[FaceBlur] Error: {e}")
        return frame


def blur_faces_black_bar(frame, violence_detected=None):
    if not FACE_DETECTOR_AVAILABLE or face_cascade is None:
        return frame
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=3, minSize=(40, 40))
        for (x, y, w, h) in faces:
            padding = int(w * 0.05)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        return frame
    except Exception as e:
        print(f"[FaceBlur] Error: {e}")
        return frame


def feed_detections_to_blurrer(camera_id, face_results=None, pose_results=None):
    """Feed pre-computed detection results into the face blurrer's cache."""
    if face_results is not None:
        _face_blurrer.yolo_results_cache[camera_id] = face_results
    elif pose_results is not None:
        _face_blurrer.yolo_results_cache[camera_id] = pose_results


__all__ = [
    'model',
    'CONF_THRESHOLD',
    'device',
    'blur_faces',
    'force_blur_faces',
    'register_violence_event',
    'reset_violence_state',
    'get_violence_status',
    'set_violence_memory_duration',
    'reset_face_tracker',
    'set_detection_mode',
    'blur_faces_pixelate',
    'blur_faces_black_bar',
    'FACE_DETECTOR_AVAILABLE',
    'HybridFaceBlurrer',
    'SmoothFaceTracker',
    'PerCameraViolenceTracker',
    'extract_face_from_pose',
    'merge_face_detections',
    'feed_detections_to_blurrer',
    'face_model',
    'FACE_CONF_THRESHOLD',
    'FACE_IOU_THRESHOLD',
    'FACE_IMG_SIZE',
]
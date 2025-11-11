"""
Reduce duplication or improvement in UID persistence 

"""


import time
import numpy as np
from scipy.spatial.distance import cosine
import torch
import torchreid
from collections import deque


class PersistentPersonTracker:

    def __init__(
        self,
        db_handler=None,
        similarity_threshold=0.70,
        cross_camera_threshold=0.75,
        min_box_area=5000,
        model_name='osnet_x1_0',
        use_gpu=True,
        max_features=50,
        confirmation_frames=2,
        db_feature_limit=10,
        use_db_for_matching=True,
        cross_camera_time_window=30.0,
        feature_failure_patience=10,
        track_memory_duration=30.0,
        spatial_proximity_threshold=150,
        spatial_proximity_bonus=0.15,
        spatial_time_window=5.0,
        temporal_smoothing_window=2.0,
        min_votes_for_smoothing=2,
        min_feature_distance=0.15,
        bbox_iou_threshold=0.3,
        simultaneous_detection_window=2.0,
        crossing_detection_enabled=True,
        crossing_spatial_threshold=200,
        crossing_time_window=3.0,
        min_feature_separation=0.20,
        spatial_bonus_decay_rate=0.6,
        motion_history_length=10,
        velocity_weight=0.15,
        bbox_overlap_penalty=0.25,
        assignment_lock_duration=1.0,
    ):
        self.db = db_handler
        self.db_feature_limit = db_feature_limit
        self.use_db_for_matching = use_db_for_matching
        
        self.persistent_ids = {}
        self.feature_history = {}
        self.track_to_persistent = {}
        self.next_persistent_id = 1
        
        self.last_seen = {}
        self.camera_locations = {}
        self.camera_history = {}
        self.spatial_context = {}
        self.first_seen = {}
        self.geo_fence_entry = {}
        
        self.similarity_threshold = similarity_threshold
        self.cross_camera_threshold = cross_camera_threshold
        self.min_box_area = min_box_area
        self.max_features_per_person = max_features
        self.confirmation_frames = confirmation_frames
        self.cross_camera_time_window = cross_camera_time_window
        self.feature_failure_patience = feature_failure_patience
        
        self.track_history = {}
        self.track_memory_duration = track_memory_duration
        
        self.spatial_proximity_threshold = spatial_proximity_threshold
        self.spatial_proximity_bonus = spatial_proximity_bonus
        self.spatial_time_window = spatial_time_window
        
        self.recent_candidates = {}
        self.temporal_smoothing_window = temporal_smoothing_window
        self.min_votes_for_smoothing = min_votes_for_smoothing
        
        self.pending_cross_matches = {}
        
        self.active_uids_per_camera = {}
        self.uid_assignment_lock = {}
        
        self.feature_extraction_failures = {}
        self.last_successful_feature = {}
        
        self.min_feature_distance = min_feature_distance
        self.bbox_iou_threshold = bbox_iou_threshold
        self.simultaneous_detection_window = simultaneous_detection_window
        self.recent_assignments = {}
        
        self.motion_history_length = motion_history_length
        self.position_history = {}
        self.velocity_estimates = {}
        self.velocity_weight = velocity_weight
        
        self.crossing_detection_enabled = crossing_detection_enabled
        self.crossing_spatial_threshold = crossing_spatial_threshold
        self.crossing_time_window = crossing_time_window
        self.min_feature_separation = min_feature_separation
        self.spatial_bonus_decay_rate = spatial_bonus_decay_rate
        self.bbox_overlap_penalty = bbox_overlap_penalty
        self.assignment_lock_duration = assignment_lock_duration
        
        self.active_crossings = {}
        self.assignment_locks = {}
        self.last_assignments = {}
        
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"[ReID] Loading TorchREID model: {model_name} on {self.device}")
        
        try:
            self.model = torchreid.models.build_model(
                name=model_name,
                num_classes=1000,
                loss='softmax',
                pretrained=True
            )
            self.model.eval()
            self.model.to(self.device)
            
            dummy_input = torch.randn(1, 3, 256, 128).to(self.device)
            with torch.no_grad():
                dummy_output = self.model(dummy_input)
            print(f"[ReID] ✅ Model initialized - Output dim: {dummy_output.shape}")
            
        except Exception as e:
            print(f"[ReID] ❌ Failed to load model: {e}")
            raise
        
        try:
            from torchvision import transforms as T
            
            self.transform = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print(f"[ReID] ✅ Transform initialized using torchvision")
        except Exception as e:
            print(f"[ReID] ⚠️ Failed to initialize torchvision transforms: {e}")
            try:
                self.transform = torchreid.data.transforms.build_transforms(
                    height=256,
                    width=128,
                    transforms=['resize', 'totensor', 'normalize']
                )
                print(f"[ReID] ✅ Transform initialized using torchreid")
            except Exception as e2:
                print(f"[ReID] ❌ Failed to initialize transforms: {e2}")
                raise
        
        if self.db and self.use_db_for_matching:
            self._load_features_from_db()
        
        print(f"[ReID] 🎯 Enhanced configuration:")
        print(f"  - Same-camera threshold: {self.similarity_threshold}")
        print(f"  - Cross-camera threshold: {self.cross_camera_threshold}")
        print(f"  - Spatial proximity: {self.spatial_proximity_threshold}px")
        print(f"  - Min feature distance: {self.min_feature_distance}")
        print(f"  - BBox IoU threshold: {self.bbox_iou_threshold}")
        print(f"[ReID] 🛡️ Anti-crossing protection:")
        print(f"  - Crossing spatial threshold: {self.crossing_spatial_threshold}px")
        print(f"  - Crossing time window: {self.crossing_time_window}s")
        print(f"  - Min feature separation: {self.min_feature_separation}")
        print(f"  - Motion history length: {self.motion_history_length}")
        print(f"  - Velocity weight: {self.velocity_weight}")

    def _load_features_from_db(self):
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT persistent_id, feature_vector, camera_id, timestamp
                    FROM reid_features
                    ORDER BY timestamp DESC
                """)
                
                rows = cursor.fetchall()
                features_by_pid = {}
                
                for row in rows:
                    pid = row[0]
                    feature_blob = row[1]
                    feature = np.frombuffer(feature_blob, dtype=np.float32)
                    
                    if pid not in features_by_pid:
                        features_by_pid[pid] = []
                    
                    if len(features_by_pid[pid]) < self.max_features_per_person:
                        features_by_pid[pid].append(feature)
                
                for pid, features in features_by_pid.items():
                    self.feature_history[pid] = features
                    self.persistent_ids[pid] = features[0]
                    
                    if pid >= self.next_persistent_id:
                        self.next_persistent_id = pid + 1
                
                print(f"[ReID] Loaded {len(features_by_pid)} persons from database")
                print(f"[ReID] Next UID will be: {self.next_persistent_id}")
                
        except Exception as e:
            print(f"[ReID] ⚠️ Failed to load database features: {e}")

    def _save_feature_to_db(self, pid, camera_id, feature, confidence=0.0):
        if not self.db or feature is None:
            return
        
        try:
            feature_np = self._normalize_feature(feature)
            if feature_np is None:
                return
            
            feature_bytes = feature_np.astype(np.float32).tobytes()
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT COUNT(*) FROM reid_features 
                    WHERE persistent_id = ?
                """, (pid,))
                
                count = cursor.fetchone()[0]
                
                if count < self.db_feature_limit:
                    cursor.execute("""
                        INSERT INTO reid_features 
                        (persistent_id, camera_id, feature_vector, confidence, 
                         extraction_method, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        pid,
                        camera_id,
                        feature_bytes,
                        confidence,
                        'osnet',
                        time.strftime('%Y-%m-%d %H:%M:%S')
                    ))
                    
        except Exception as e:
            print(f"[ReID] ⚠️ Failed to save feature to DB: {e}")

    def extract_feature(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return None
            
        try:
            from PIL import Image
            
            if len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
                if image_crop.dtype != np.uint8:
                    image_crop = np.clip(image_crop, 0, 255).astype(np.uint8)
                image_rgb = Image.fromarray(image_crop[:, :, ::-1])
            else:
                if image_crop.dtype != np.uint8:
                    image_crop = np.clip(image_crop, 0, 255).astype(np.uint8)
                image_rgb = Image.fromarray(image_crop)
            
            if image_rgb.mode != 'RGB':
                image_rgb = image_rgb.convert('RGB')
            
            if hasattr(self.transform, '__call__'):
                img_tensor = self.transform(image_rgb)
            else:
                img_tensor = image_rgb
                for t in self.transform:
                    img_tensor = t(img_tensor)
            
            if not isinstance(img_tensor, torch.Tensor):
                raise ValueError(f"Transform did not return a tensor")
            
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            features = features.cpu().numpy().flatten()
            norm = np.linalg.norm(features)
            
            return features / norm if norm > 0 else None
            
        except Exception as e:
            print(f"[ReID] ⚠️ Feature extraction failed: {e}")
            return None

    def _normalize_feature(self, feature):
        if feature is None:
            return None
            
        if isinstance(feature, np.ndarray):
            feat_np = feature
        elif hasattr(feature, 'is_cuda') and feature.is_cuda:
            feat_np = feature.detach().cpu().numpy()
        elif hasattr(feature, 'numpy'):
            feat_np = feature.numpy()
        else:
            feat_np = np.array(feature)
            
        feat_np = feat_np.flatten()
        norm = np.linalg.norm(feat_np)
        
        return feat_np / norm if norm > 0 else feat_np

    def _is_valid_detection(self, bbox):
        if bbox is None or len(bbox) != 4:
            return False
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return area >= self.min_box_area

    def _calculate_similarity(self, feature1, feature2):
        feat1 = self._normalize_feature(feature1)
        feat2 = self._normalize_feature(feature2)
        
        if feat1 is None or feat2 is None:
            return 0.0
            
        return 1 - cosine(feat1, feat2)

    def _add_feature_to_history(self, pid, feature):
        if feature is None:
            return
            
        feature_np = self._normalize_feature(feature)
        
        if pid not in self.feature_history:
            self.feature_history[pid] = []
        
        is_diverse = True
        for existing_feature in self.feature_history[pid]:
            similarity = self._calculate_similarity(feature_np, existing_feature)
            if similarity > 0.92:
                is_diverse = False
                break
        
        if is_diverse:
            if len(self.feature_history[pid]) >= self.max_features_per_person:
                self.feature_history[pid].pop(0)
            self.feature_history[pid].append(feature_np)
            self.persistent_ids[pid] = feature_np

    def _calculate_spatial_distance(self, bbox1, bbox2):
        if bbox1 is None or bbox2 is None:
            return float('inf')
        
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        distance = np.sqrt(
            (center1[0] - center2[0])**2 + 
            (center1[1] - center2[1])**2
        )
        
        return distance

    def _calculate_bbox_iou(self, bbox1, bbox2):
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _update_motion_history(self, pid, bbox, timestamp):
        if bbox is None:
            return
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        if pid not in self.position_history:
            self.position_history[pid] = deque(maxlen=self.motion_history_length)
        
        self.position_history[pid].append((center_x, center_y, timestamp))
        
        if len(self.position_history[pid]) >= 2:
            recent = list(self.position_history[pid])[-5:]
            if len(recent) >= 2:
                dt = recent[-1][2] - recent[0][2]
                if dt > 0:
                    dx = recent[-1][0] - recent[0][0]
                    dy = recent[-1][1] - recent[0][1]
                    self.velocity_estimates[pid] = (dx/dt, dy/dt)

    def _predict_position(self, pid, time_delta):
        if pid not in self.velocity_estimates or pid not in self.position_history:
            return None
        
        if not self.position_history[pid]:
            return None
        
        last_pos = self.position_history[pid][-1]
        vx, vy = self.velocity_estimates[pid]
        
        pred_x = last_pos[0] + vx * time_delta
        pred_y = last_pos[1] + vy * time_delta
        
        return (pred_x, pred_y)

    def _calculate_motion_consistency(self, pid, new_bbox, timestamp):
        if pid not in self.position_history or not self.position_history[pid]:
            return 0.5
        
        last_pos = self.position_history[pid][-1]
        time_delta = timestamp - last_pos[2]
        
        if time_delta <= 0:
            return 0.5
        
        predicted = self._predict_position(pid, time_delta)
        if predicted is None:
            return 0.5
        
        actual_x = (new_bbox[0] + new_bbox[2]) / 2
        actual_y = (new_bbox[1] + new_bbox[3]) / 2
        
        distance = np.sqrt((actual_x - predicted[0])**2 + (actual_y - predicted[1])**2)
        
        max_expected_movement = 500 * time_delta
        
        if distance > max_expected_movement:
            return 0.0
        
        score = max(0.0, 1.0 - (distance / max_expected_movement))
        
        return score

    def _detect_crossing_batch(self, camera_id, frame_state):
        if not self.crossing_detection_enabled:
            return set()
        
        crossing_pairs = set()
        current_time = time.time()
        
        pids = list(frame_state.keys())
        
        for i, pid1 in enumerate(pids):
            for pid2 in pids[i+1:]:
                bbox1, feat1 = frame_state[pid1]
                bbox2, feat2 = frame_state[pid2]
                
                distance = self._calculate_spatial_distance(bbox1, bbox2)
                
                if distance < self.crossing_spatial_threshold:
                    feature_sim = self._calculate_similarity(feat1, feat2)
                    feature_diff = 1.0 - feature_sim
                    
                    if feature_diff < self.min_feature_separation:
                        pair = tuple(sorted([pid1, pid2]))
                        crossing_pairs.add(pair)
                        
                        print(f"[ReID] ⚠️ CROSSING DETECTED: PID {pid1} ↔ PID {pid2} "
                              f"(dist: {distance:.0f}px, similarity: {feature_sim:.3f})")
        
        if camera_id not in self.active_crossings:
            self.active_crossings[camera_id] = {}
        
        for pair in crossing_pairs:
            if pair not in self.active_crossings[camera_id]:
                self.active_crossings[camera_id][pair] = {
                    'start_time': current_time,
                    'last_seen': current_time,
                    'locked': True
                }
            else:
                self.active_crossings[camera_id][pair]['last_seen'] = current_time
        
        expired = []
        for pair, info in list(self.active_crossings[camera_id].items()):
            if current_time - info['last_seen'] > self.crossing_time_window:
                expired.append(pair)
                print(f"[ReID] ✅ CROSSING RESOLVED: PID {pair[0]} ↔ PID {pair[1]}")
        
        for pair in expired:
            del self.active_crossings[camera_id][pair]
        
        return crossing_pairs

    def _is_crossing_active(self, camera_id, pid):
        if camera_id not in self.active_crossings:
            return False
        
        for pair in self.active_crossings[camera_id].keys():
            if pid in pair:
                return True
        
        return False

    def _check_assignment_conflicts(self, camera_id, candidate_pid, new_bbox, new_feature):
        current_time = time.time()
        
        if camera_id not in self.last_assignments:
            return False, None, 0.0
        
        for pid, (old_bbox, old_feature, timestamp) in self.last_assignments[camera_id].items():
            if pid != candidate_pid:
                continue
            
            time_diff = current_time - timestamp
            
            iou = self._calculate_bbox_iou(new_bbox, old_bbox)
            
            if iou < 0.2:
                feature_sim = self._calculate_similarity(new_feature, old_feature)
                motion_score = self._calculate_motion_consistency(candidate_pid, new_bbox, current_time)
                
                if feature_sim > 0.75 and motion_score < 0.3:
                    reason = (f"PID {pid} position jump detected "
                             f"(IoU: {iou:.2f}, motion: {motion_score:.2f})")
                    return True, reason, 0.3
                
                elif feature_sim < 0.65:
                    reason = (f"PID {pid} feature mismatch "
                             f"(similarity: {feature_sim:.2f})")
                    return True, reason, 0.5
            
            if self._is_crossing_active(camera_id, pid):
                feature_sim = self._calculate_similarity(new_feature, old_feature)
                
                if feature_sim < 0.75:
                    reason = f"PID {pid} uncertain during crossing (sim: {feature_sim:.2f})"
                    return True, reason, 0.4
                
                return False, None, 0.15
        
        return False, None, 0.0

    def _apply_crossing_penalties(self, camera_id, pid, base_score, bbox):
        if not self._is_crossing_active(camera_id, pid):
            return base_score
        
        crossing_pair = None
        for pair in self.active_crossings[camera_id].keys():
            if pid in pair:
                crossing_pair = pair
                break
        
        if crossing_pair is None:
            return base_score
        
        crossing_info = self.active_crossings[camera_id][crossing_pair]
        time_in_crossing = time.time() - crossing_info['start_time']
        
        spatial_decay = min(1.0, time_in_crossing / self.crossing_time_window)
        penalty = self.spatial_bonus_decay_rate * spatial_decay
        
        adjusted_score = base_score * (1.0 - penalty)
        
        print(f"[ReID] 🔻 Crossing penalty: PID {pid} "
              f"({base_score:.3f} → {adjusted_score:.3f}, -{penalty*100:.1f}%)")
        
        return adjusted_score

    def _check_duplicate_assignment(self, camera_id, candidate_pid, new_feature, new_bbox):
        if camera_id not in self.recent_assignments:
            return False, None
        
        current_time = time.time()
        
        self.recent_assignments[camera_id] = [
            (pid, feat, bbox, ts) for pid, feat, bbox, ts in self.recent_assignments[camera_id]
            if current_time - ts < self.simultaneous_detection_window
        ]
        
        for pid, existing_feat, existing_bbox, timestamp in self.recent_assignments[camera_id]:
            if pid != candidate_pid:
                continue
            
            time_diff = current_time - timestamp
            
            if new_bbox is not None and existing_bbox is not None:
                iou = self._calculate_bbox_iou(new_bbox, existing_bbox)
                
                if iou < self.bbox_iou_threshold:
                    feature_sim = self._calculate_similarity(new_feature, existing_feat)
                    
                    if feature_sim > self.similarity_threshold:
                        reason = (f"PID {pid} already assigned to different person "
                                 f"(IoU: {iou:.2f}, similarity: {feature_sim:.2f}, "
                                 f"time_diff: {time_diff:.2f}s)")
                        return True, reason
            
            if new_feature is not None and existing_feat is not None:
                feature_sim = self._calculate_similarity(new_feature, existing_feat)
                
                if feature_sim < (self.similarity_threshold - self.min_feature_distance):
                    reason = (f"PID {pid} feature mismatch "
                             f"(similarity: {feature_sim:.2f}, expected > {self.similarity_threshold:.2f})")
                    return True, reason
        
        return False, None

    def _record_assignment(self, camera_id, pid, feature, bbox):
        if camera_id not in self.recent_assignments:
            self.recent_assignments[camera_id] = []
        
        current_time = time.time()
        self.recent_assignments[camera_id].append((pid, feature, bbox, current_time))
        
        self.recent_assignments[camera_id] = [
            (p, f, b, t) for p, f, b, t in self.recent_assignments[camera_id]
            if current_time - t < self.simultaneous_detection_window
        ]

    def process_frame_batch(self, camera_id, detections):
        current_time = time.time()
        results = []
        
        frame_state = {}
        new_detections = []
        
        for track_id, bbox, feature, in_geo_fence in detections:
            track_key = (camera_id, track_id)
            
            if track_key in self.track_to_persistent:
                pid = self.track_to_persistent[track_key]
                frame_state[pid] = (bbox, feature if feature is not None else self.persistent_ids.get(pid))
            else:
                new_detections.append((track_id, bbox, feature, in_geo_fence))
        
        if len(frame_state) > 1:
            self._detect_crossing_batch(camera_id, frame_state)
        
        for track_id, bbox, feature, in_geo_fence in detections:
            track_key = (camera_id, track_id)
            
            if track_key in self.track_to_persistent:
                pid = self.track_to_persistent[track_key]
                
                if feature is not None:
                    self._add_feature_to_history(pid, feature)
                    self.last_successful_feature[track_key] = (feature, current_time)
                    self.feature_extraction_failures[track_key] = 0
                    self._record_assignment(camera_id, pid, feature, bbox)
                else:
                    self.feature_extraction_failures[track_key] = \
                        self.feature_extraction_failures.get(track_key, 0) + 1
                
                self.last_seen[pid] = current_time
                self.camera_locations[pid] = camera_id
                
                if bbox is not None:
                    self.spatial_context[pid] = bbox
                    self._update_motion_history(pid, bbox, current_time)
                
                if pid not in self.camera_history:
                    self.camera_history[pid] = set()
                self.camera_history[pid].add(camera_id)
                
                if in_geo_fence and pid not in self.geo_fence_entry:
                    self.geo_fence_entry[pid] = current_time
                
                if camera_id not in self.active_uids_per_camera:
                    self.active_uids_per_camera[camera_id] = set()
                self.active_uids_per_camera[camera_id].add(pid)
                
                results.append((track_id, pid))
        
        for track_id, bbox, feature, in_geo_fence in new_detections:
            pid = self.get_or_create_persistent_id(
                camera_id, track_id, feature, bbox, in_geo_fence
            )
            
            if pid is not None:
                frame_state[pid] = (bbox, feature)
                results.append((track_id, pid))
        
        return results

    def find_best_match(self, camera_id, feature, bbox=None, track_id=None):
        if feature is None:
            return None, 0.0
            
        if bbox is not None and not self._is_valid_detection(bbox):
            return None, 0.0

        feature_np = self._normalize_feature(feature)
        current_time = time.time()
        
        candidates = []

        for pid, features_list in self.feature_history.items():
            if not features_list:
                continue
            
            similarities = [self._calculate_similarity(feature_np, f) for f in features_list]
            max_similarity = max(similarities)
            avg_similarity = np.mean(similarities)
            base_similarity = 0.7 * max_similarity + 0.3 * avg_similarity
            
            motion_score = 0.5
            motion_bonus = 0.0
            if bbox is not None:
                motion_score = self._calculate_motion_consistency(pid, bbox, current_time)
                motion_bonus = self.velocity_weight * motion_score
            
            is_same_camera = (pid in self.camera_locations and 
                             self.camera_locations[pid] == camera_id)
            
            spatial_bonus = 0.0
            if is_same_camera and bbox is not None and pid in self.spatial_context:
                time_elapsed = current_time - self.last_seen.get(pid, current_time)
                
                if time_elapsed < self.spatial_time_window:
                    distance = self._calculate_spatial_distance(bbox, self.spatial_context[pid])
                    
                    if distance < self.spatial_proximity_threshold:
                        spatial_factor = 1 - (distance / self.spatial_proximity_threshold)
                        spatial_bonus = self.spatial_proximity_bonus * spatial_factor
                        
                        if distance < 30:
                            spatial_bonus += 0.05
                        
                        print(f"[ReID] 📍 Spatial bonus: PID {pid} "
                              f"(dist: {distance:.0f}px, bonus: +{spatial_bonus*100:.1f}%)")
            
            combined_score = base_similarity + motion_bonus + spatial_bonus
            
            if bbox is not None:
                combined_score = self._apply_crossing_penalties(camera_id, pid, combined_score, bbox)
            
            if bbox is not None:
                is_conflict, reason, penalty = self._check_assignment_conflicts(
                    camera_id, pid, bbox, feature_np
                )
                
                if is_conflict:
                    print(f"[ReID] ❌ Conflict: {reason}")
                    continue
                
                combined_score -= penalty
            
            if bbox is not None:
                is_duplicate, reason = self._check_duplicate_assignment(
                    camera_id, pid, feature_np, bbox
                )
                
                if is_duplicate:
                    print(f"[ReID] ⚠️ DUPLICATE PREVENTED: {reason}")
                    continue
            
            threshold = self.similarity_threshold if is_same_camera else self.cross_camera_threshold
            
            if combined_score >= threshold:
                candidates.append((pid, combined_score, is_same_camera, motion_score))
        
        if not candidates:
            return None, 0.0
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_pid, best_score, is_same_cam, motion = candidates[0]
        
        if track_id is not None:
            track_key = (camera_id, track_id)
            
            if track_key not in self.recent_candidates:
                self.recent_candidates[track_key] = []
            
            self.recent_candidates[track_key].append({
                'pid': best_pid,
                'score': best_score,
                'time': current_time
            })
            
            self.recent_candidates[track_key] = [
                c for c in self.recent_candidates[track_key]
                if current_time - c['time'] < self.temporal_smoothing_window
            ]
            
            if len(self.recent_candidates[track_key]) >= self.min_votes_for_smoothing:
                pid_votes = {}
                for candidate in self.recent_candidates[track_key]:
                    pid = candidate['pid']
                    pid_votes[pid] = pid_votes.get(pid, 0) + candidate['score']
                
                voted_pid = max(pid_votes, key=pid_votes.get)
                voted_score = pid_votes[voted_pid] / len(self.recent_candidates[track_key])
                
                if voted_pid != best_pid:
                    if bbox is not None:
                        is_dup, _ = self._check_duplicate_assignment(
                            camera_id, voted_pid, feature_np, bbox
                        )
                        
                        if not is_dup:
                            print(f"[ReID] 🗳️ Smoothing: {best_pid} -> {voted_pid}")
                            best_pid = voted_pid
                            best_score = voted_score
        
        if bbox is not None:
            self._update_motion_history(best_pid, bbox, current_time)
        
        if bbox is not None:
            if camera_id not in self.last_assignments:
                self.last_assignments[camera_id] = {}
            
            self.last_assignments[camera_id][best_pid] = (bbox, feature_np, current_time)
            
            expired = [
                pid for pid, (_, _, ts) in self.last_assignments[camera_id].items()
                if current_time - ts > 2.0
            ]
            for pid in expired:
                del self.last_assignments[camera_id][pid]
        
        match_type = "same-cam" if is_same_cam else "cross-cam"
        print(f"[ReID] ✓ Match: PID {best_pid} ({match_type}, "
              f"score: {best_score:.3f}, motion: {motion:.2f})")
        
        return best_pid, best_score

    def get_or_create_persistent_id(self, camera_id, track_id, feature, bbox=None, in_geo_fence=False):
        if bbox is not None and not self._is_valid_detection(bbox):
            return None

        track_key = (camera_id, track_id)
        current_time = time.time()

        if camera_id not in self.active_uids_per_camera:
            self.active_uids_per_camera[camera_id] = set()

        if track_key in self.track_history:
            old_pid, last_seen = self.track_history[track_key]
            time_since_lost = current_time - last_seen
            
            if time_since_lost < self.track_memory_duration:
                self.track_to_persistent[track_key] = old_pid
                
                if feature is not None:
                    self._add_feature_to_history(old_pid, feature)
                    self.last_successful_feature[track_key] = (feature, current_time)
                    self.feature_extraction_failures[track_key] = 0
                    if bbox is not None:
                        self._record_assignment(camera_id, old_pid, feature, bbox)
                
                self.last_seen[old_pid] = current_time
                self.camera_locations[old_pid] = camera_id
                
                if bbox is not None:
                    self.spatial_context[old_pid] = bbox
                    self._update_motion_history(old_pid, bbox, current_time)
                
                if old_pid not in self.camera_history:
                    self.camera_history[old_pid] = set()
                self.camera_history[old_pid].add(camera_id)
                
                self.active_uids_per_camera[camera_id].add(old_pid)
                
                del self.track_history[track_key]
                
                print(f"[ReID] ✅ Resumed UID {old_pid} for track {track_id} "
                      f"(was lost for {time_since_lost:.1f}s)")
                return old_pid

        if track_key in self.track_to_persistent:
            pid = self.track_to_persistent[track_key]
            
            if feature is not None:
                self._add_feature_to_history(pid, feature)
                self.last_successful_feature[track_key] = (feature, current_time)
                self.feature_extraction_failures[track_key] = 0
                if bbox is not None:
                    self._record_assignment(camera_id, pid, feature, bbox)
            else:
                self.feature_extraction_failures[track_key] = \
                    self.feature_extraction_failures.get(track_key, 0) + 1
            
            self.last_seen[pid] = current_time
            self.camera_locations[pid] = camera_id
            
            if bbox is not None:
                self.spatial_context[pid] = bbox
                self._update_motion_history(pid, bbox, current_time)
            
            if pid not in self.camera_history:
                self.camera_history[pid] = set()
            self.camera_history[pid].add(camera_id)
            
            if in_geo_fence and pid not in self.geo_fence_entry:
                self.geo_fence_entry[pid] = current_time
                print(f"[ReID] Person {pid} entered geo-fence in {camera_id}")
            
            self.active_uids_per_camera[camera_id].add(pid)
            
            return pid

        if feature is None:
            if track_key in self.last_successful_feature:
                last_feat, last_time = self.last_successful_feature[track_key]
                if current_time - last_time < 2.0:
                    feature = last_feat
            
            self.feature_extraction_failures[track_key] = \
                self.feature_extraction_failures.get(track_key, 0) + 1
            
            failure_count = self.feature_extraction_failures[track_key]
            
            if failure_count >= self.feature_failure_patience:
                print(f"[ReID] ⚠️ Creating UID without feature after {failure_count} failures")
            elif feature is None:
                return None

        if feature is not None:
            match_id, confidence = self.find_best_match(camera_id, feature, bbox, track_id)
            
            if match_id:
                is_cross_camera = (match_id in self.camera_locations and 
                                  self.camera_locations[match_id] != camera_id)
                
                if is_cross_camera:
                    pending_key = track_key
                    
                    if pending_key not in self.pending_cross_matches:
                        self.pending_cross_matches[pending_key] = {
                            'pid': match_id,
                            'scores': [confidence],
                            'features': [feature],
                            'count': 1,
                            'failures': 0,
                            'first_seen': current_time
                        }
                        print(f"[ReID] 🔄 Pending cross-camera: Track {track_id} -> PID {match_id} "
                              f"(1/{self.confirmation_frames}, score: {confidence:.3f})")
                        return None
                    else:
                        pending = self.pending_cross_matches[pending_key]
                        
                        if pending['pid'] != match_id:
                            if confidence > np.mean(pending['scores']):
                                print(f"[ReID] 🔄 Cross-camera improved: {pending['pid']} -> {match_id}")
                                pending['pid'] = match_id
                                pending['scores'] = [confidence]
                                pending['features'] = [feature]
                                pending['count'] = 1
                                pending['failures'] = 0
                            else:
                                print(f"[ReID] ⚠️ Cross-camera inconsistent, keeping {pending['pid']}")
                            return None
                        
                        pending['scores'].append(confidence)
                        pending['features'].append(feature)
                        pending['count'] += 1
                        pending['failures'] = 0
                        
                        print(f"[ReID] 🔄 Pending cross-camera: Track {track_id} -> PID {match_id} "
                              f"({pending['count']}/{self.confirmation_frames}, avg: {np.mean(pending['scores']):.3f})")
                        
                        if pending['count'] >= self.confirmation_frames:
                            avg_confidence = np.mean(pending['scores'])
                            
                            if avg_confidence >= self.cross_camera_threshold:
                                self.track_to_persistent[track_key] = match_id
                                self._add_feature_to_history(match_id, feature)
                                self.last_seen[match_id] = current_time
                                self.camera_locations[match_id] = camera_id
                                
                                if bbox is not None:
                                    self.spatial_context[match_id] = bbox
                                    self._update_motion_history(match_id, bbox, current_time)
                                
                                if match_id not in self.camera_history:
                                    self.camera_history[match_id] = set()
                                self.camera_history[match_id].add(camera_id)
                                
                                self._save_feature_to_db(match_id, camera_id, feature, avg_confidence)
                                
                                self.active_uids_per_camera[camera_id].add(match_id)
                                self.uid_assignment_lock[match_id] = camera_id
                                
                                self.feature_extraction_failures[track_key] = 0
                                self.last_successful_feature[track_key] = (feature, current_time)
                                
                                if bbox is not None:
                                    self._record_assignment(camera_id, match_id, feature, bbox)
                                
                                del self.pending_cross_matches[pending_key]
                                
                                print(f"[ReID] ✅ CONFIRMED cross-camera: Track {track_id} -> PID {match_id} "
                                      f"(avg: {avg_confidence:.3f})")
                                return match_id
                            else:
                                print(f"[ReID] ❌ Cross-camera REJECTED (avg: {avg_confidence:.3f})")
                                del self.pending_cross_matches[pending_key]
                        else:
                            return None
                else:
                    if confidence >= self.similarity_threshold:
                        is_dup, reason = self._check_duplicate_assignment(
                            camera_id, match_id, feature, bbox
                        )
                        
                        if is_dup:
                            print(f"[ReID] ❌ Match rejected (duplicate): {reason}")
                            match_id = None
                        else:
                            self.track_to_persistent[track_key] = match_id
                            self._add_feature_to_history(match_id, feature)
                            self.last_seen[match_id] = current_time
                            self.camera_locations[match_id] = camera_id
                            
                            if bbox is not None:
                                self.spatial_context[match_id] = bbox
                                self._update_motion_history(match_id, bbox, current_time)
                            
                            if match_id not in self.camera_history:
                                self.camera_history[match_id] = set()
                            self.camera_history[match_id].add(camera_id)
                            
                            self.active_uids_per_camera[camera_id].add(match_id)
                            
                            self.feature_extraction_failures[track_key] = 0
                            self.last_successful_feature[track_key] = (feature, current_time)
                            
                            if bbox is not None:
                                self._record_assignment(camera_id, match_id, feature, bbox)
                            
                            print(f"[ReID] ✅ Same-camera match: Track {track_id} -> PID {match_id} "
                                  f"(confidence: {confidence:.3f})")
                            return match_id

        new_pid = self.next_persistent_id
        self.next_persistent_id += 1
        
        self.track_to_persistent[track_key] = new_pid
        self.feature_history[new_pid] = [feature] if feature is not None else []
        self.persistent_ids[new_pid] = feature
        self.last_seen[new_pid] = current_time
        self.first_seen[new_pid] = current_time
        self.camera_locations[new_pid] = camera_id
        self.camera_history[new_pid] = {camera_id}
        
        if bbox is not None:
            self.spatial_context[new_pid] = bbox
            self._update_motion_history(new_pid, bbox, current_time)
        
        if in_geo_fence:
            self.geo_fence_entry[new_pid] = current_time
        
        self.active_uids_per_camera[camera_id].add(new_pid)
        self.uid_assignment_lock[new_pid] = camera_id
        
        self.feature_extraction_failures[track_key] = 0
        if feature is not None:
            self.last_successful_feature[track_key] = (feature, current_time)
            if bbox is not None:
                self._record_assignment(camera_id, new_pid, feature, bbox)
        
        if feature is not None:
            self._save_feature_to_db(new_pid, camera_id, feature, 1.0)
        
        print(f"[ReID] 🆕 NEW UID: {new_pid} for track {track_id} in {camera_id}")
        return new_pid

    def cleanup_old_tracks(self, camera_id, active_track_ids, max_age=300):
        camera_tracks = {k: v for k, v in self.track_to_persistent.items() 
                        if k[0] == camera_id}
        
        active_keys = {(camera_id, tid) for tid in active_track_ids}
        inactive_tracks = set(camera_tracks.keys()) - active_keys
        
        current_time = time.time()
        
        for track_key in inactive_tracks:
            pid = self.track_to_persistent[track_key]
            
            self.track_history[track_key] = (pid, current_time)
            print(f"[ReID] 💾 Stored track {track_key[1]} -> PID {pid} in memory "
                  f"(will expire in {self.track_memory_duration}s)")
            
            if camera_id in self.active_uids_per_camera:
                self.active_uids_per_camera[camera_id].discard(pid)
            
            del self.track_to_persistent[track_key]
            
            if track_key in self.pending_cross_matches:
                del self.pending_cross_matches[track_key]
            
            self.feature_extraction_failures.pop(track_key, None)
            self.last_successful_feature.pop(track_key, None)
            self.recent_candidates.pop(track_key, None)

        expired_history = [
            k for k, (pid, last_seen) in self.track_history.items()
            if current_time - last_seen > self.track_memory_duration
        ]
        
        for track_key in expired_history:
            pid, last_seen = self.track_history[track_key]
            age = current_time - last_seen
            del self.track_history[track_key]
            print(f"[ReID] 🗑️ Expired track memory: {track_key} -> PID {pid} (age: {age:.1f}s)")

        old_pids = [pid for pid, last_time in self.last_seen.items() 
                   if current_time - last_time > max_age]
        
        for pid in old_pids:
            for cam_id in self.active_uids_per_camera:
                self.active_uids_per_camera[cam_id].discard(pid)
            
            self.persistent_ids.pop(pid, None)
            self.feature_history.pop(pid, None)
            self.last_seen.pop(pid, None)
            self.camera_locations.pop(pid, None)
            self.spatial_context.pop(pid, None)
            self.camera_history.pop(pid, None)
            self.uid_assignment_lock.pop(pid, None)
            self.position_history.pop(pid, None)
            self.velocity_estimates.pop(pid, None)
            
            print(f"[ReID] 🗑️ Removed PID {pid} from memory (age: {max_age}s)")

    def get_active_uids_in_camera(self, camera_id):
        return self.active_uids_per_camera.get(camera_id, set())

    def get_statistics(self):
        total_active = sum(len(uids) for uids in self.active_uids_per_camera.values())
        total_crossings = sum(len(crossings) for crossings in self.active_crossings.values())
        
        return {
            'total_persistent_ids': len(self.persistent_ids),
            'active_tracks': len(self.track_to_persistent),
            'pending_cross_matches': len(self.pending_cross_matches),
            'track_history_size': len(self.track_history),
            'temporal_candidates': len(self.recent_candidates),
            'recent_assignments': sum(len(v) for v in self.recent_assignments.values()),
            'next_id': self.next_persistent_id,
            'active_uids_total': total_active,
            'active_per_camera': {cam: len(uids) for cam, uids in self.active_uids_per_camera.items()},
            'db_enabled': self.db is not None,
            'feature_failures': len(self.feature_extraction_failures),
            'active_crossings': total_crossings,
            'motion_tracked_pids': len(self.position_history),
            'thresholds': {
                'same_camera': self.similarity_threshold,
                'cross_camera': self.cross_camera_threshold,
                'spatial_proximity': self.spatial_proximity_threshold,
                'track_memory': self.track_memory_duration,
                'bbox_iou': self.bbox_iou_threshold,
                'min_feature_distance': self.min_feature_distance,
                'crossing_spatial': self.crossing_spatial_threshold,
                'min_feature_separation': self.min_feature_separation
            }
        }

    def print_status(self):
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("📊 ENHANCED PERSISTENT PERSON TRACKER STATUS")
        print("="*60)
        print(f"Total Persons: {stats['total_persistent_ids']}")
        print(f"Active Tracks: {stats['active_tracks']}")
        print(f"Pending Cross-Camera: {stats['pending_cross_matches']}")
        print(f"Track History: {stats['track_history_size']} (memory: {self.track_memory_duration}s)")
        print(f"Temporal Candidates: {stats['temporal_candidates']}")
        print(f"Recent Assignments: {stats['recent_assignments']}")
        print(f"Active Crossings: {stats['active_crossings']}")
        print(f"Motion Tracked PIDs: {stats['motion_tracked_pids']}")
        print(f"Next UID: {stats['next_id']}")
        print(f"Feature Failures: {stats['feature_failures']}")
        print(f"\n🎯 Thresholds:")
        print(f"  Same Camera: {stats['thresholds']['same_camera']:.2f}")
        print(f"  Cross Camera: {stats['thresholds']['cross_camera']:.2f}")
        print(f"  Spatial Proximity: {stats['thresholds']['spatial_proximity']}px")
        print(f"  Track Memory: {stats['thresholds']['track_memory']:.1f}s")
        print(f"  BBox IoU: {stats['thresholds']['bbox_iou']:.2f}")
        print(f"  Min Feature Distance: {stats['thresholds']['min_feature_distance']:.2f}")
        print(f"  Crossing Spatial: {stats['thresholds']['crossing_spatial']}px")
        print(f"  Min Feature Separation: {stats['thresholds']['min_feature_separation']:.2f}")
        print(f"\n📹 Active UIDs per Camera:")
        for cam_id, uids in stats['active_per_camera'].items():
            print(f"  {cam_id}: {uids} UIDs")
        print("="*60 + "\n")

    def print_crossing_status(self):
        if not any(self.active_crossings.values()):
            return
        
        print("\n" + "="*60)
        print("⚠️  ACTIVE CROSSINGS")
        print("="*60)
        
        for camera_id, crossings in self.active_crossings.items():
            if crossings:
                print(f"\n📹 {camera_id}:")
                for pair, info in crossings.items():
                    duration = time.time() - info['start_time']
                    print(f"  PID {pair[0]} ↔ PID {pair[1]} "
                          f"(duration: {duration:.1f}s, locked: {info['locked']})")
        
        print("="*60 + "\n")
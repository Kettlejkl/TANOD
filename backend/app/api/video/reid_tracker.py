import time
import numpy as np
from scipy.spatial.distance import cosine
import torch
import torchreid
from collections import deque
import cv2


class PersistentPersonTracker:

    def __init__(
        self,
        db_handler=None,
        similarity_threshold=0.65,
        cross_camera_threshold=0.70,
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
        color_weight=0.20,
        color_bins=16,
        color_similarity_threshold=0.75,
        color_update_rate=0.3,
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
        
        self.color_weight = color_weight
        self.color_bins = color_bins
        self.color_similarity_threshold = color_similarity_threshold
        self.color_update_rate = color_update_rate
        
        self.color_histograms = {}
        self.color_confidence = {}
        
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
        print(f"[ReID] 🎨 Clothing color analysis:")
        print(f"  - Color weight: {self.color_weight}")
        print(f"  - Color bins: {self.color_bins}")
        print(f"  - Color threshold: {self.color_similarity_threshold}")
        print(f"  - Color update rate: {self.color_update_rate}")

    def _extract_color_histogram(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return None
        
        try:
            if len(image_crop.shape) != 3 or image_crop.shape[2] != 3:
                return None
            
            if image_crop.dtype != np.uint8:
                image_crop = np.clip(image_crop, 0, 255).astype(np.uint8)
            
            height, width = image_crop.shape[:2]
            
            upper_body = image_crop[:int(height*0.6), :]
            lower_body = image_crop[int(height*0.4):, :]
            
            hsv_upper = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
            hsv_lower = cv2.cvtColor(lower_body, cv2.COLOR_BGR2HSV)
            
            mask_upper = cv2.inRange(hsv_upper, np.array([0, 30, 30]), np.array([180, 255, 255]))
            mask_lower = cv2.inRange(hsv_lower, np.array([0, 30, 30]), np.array([180, 255, 255]))
            
            hist_upper_h = cv2.calcHist([hsv_upper], [0], mask_upper, [self.color_bins], [0, 180])
            hist_upper_s = cv2.calcHist([hsv_upper], [1], mask_upper, [self.color_bins], [0, 256])
            
            hist_lower_h = cv2.calcHist([hsv_lower], [0], mask_lower, [self.color_bins], [0, 180])
            hist_lower_s = cv2.calcHist([hsv_lower], [1], mask_lower, [self.color_bins], [0, 256])
            
            hist_upper_h = cv2.normalize(hist_upper_h, hist_upper_h).flatten()
            hist_upper_s = cv2.normalize(hist_upper_s, hist_upper_s).flatten()
            hist_lower_h = cv2.normalize(hist_lower_h, hist_lower_h).flatten()
            hist_lower_s = cv2.normalize(hist_lower_s, hist_lower_s).flatten()
            
            color_signature = {
                'upper_hue': hist_upper_h,
                'upper_sat': hist_upper_s,
                'lower_hue': hist_lower_h,
                'lower_sat': hist_lower_s,
            }
            
            return color_signature
            
        except Exception as e:
            print(f"[ReID] ⚠️ Color extraction failed: {e}")
            return None

    def _calculate_color_similarity(self, color1, color2):
        if color1 is None or color2 is None:
            return 0.0
        
        try:
            sim_upper_h = cv2.compareHist(color1['upper_hue'], color2['upper_hue'], cv2.HISTCMP_CORREL)
            sim_upper_s = cv2.compareHist(color1['upper_sat'], color2['upper_sat'], cv2.HISTCMP_CORREL)
            sim_lower_h = cv2.compareHist(color1['lower_hue'], color2['lower_hue'], cv2.HISTCMP_CORREL)
            sim_lower_s = cv2.compareHist(color1['lower_sat'], color2['lower_sat'], cv2.HISTCMP_CORREL)
            
            upper_score = 0.6 * sim_upper_h + 0.4 * sim_upper_s
            lower_score = 0.6 * sim_lower_h + 0.4 * sim_lower_s
            
            combined_score = 0.6 * upper_score + 0.4 * lower_score
            
            combined_score = max(0.0, min(1.0, combined_score))
            
            return combined_score
            
        except Exception as e:
            print(f"[ReID] ⚠️ Color comparison failed: {e}")
            return 0.0

    def _update_color_histogram(self, pid, new_color):
        if new_color is None:
            return
        
        if pid not in self.color_histograms:
            self.color_histograms[pid] = new_color
            self.color_confidence[pid] = 1.0
        else:
            old_color = self.color_histograms[pid]
            
            for key in ['upper_hue', 'upper_sat', 'lower_hue', 'lower_sat']:
                old_color[key] = (1 - self.color_update_rate) * old_color[key] + \
                                 self.color_update_rate * new_color[key]
            
            self.color_histograms[pid] = old_color
            self.color_confidence[pid] = min(1.0, self.color_confidence[pid] + 0.1)

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
                
                if feature_sim < self.min_feature_distance:
                    reason = (f"PID {pid} features too similar "
                             f"(similarity: {feature_sim:.2f})")
                    return True, reason
        
        return False, None

    def _record_assignment(self, camera_id, pid, feature, bbox, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        
        if camera_id not in self.recent_assignments:
            self.recent_assignments[camera_id] = []
        
        self.recent_assignments[camera_id].append((pid, feature, bbox, timestamp))
        
        if camera_id not in self.last_assignments:
            self.last_assignments[camera_id] = {}
        
        self.last_assignments[camera_id][pid] = (bbox, feature, timestamp)

    def process_detections(self, camera_id, detections, frame, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        
        results = []
        current_time = timestamp
        
        frame_state = {}
        
        for detection in detections:
            bbox = detection.get('bbox')
            track_id = detection.get('track_id')
            
            if not self._is_valid_detection(bbox):
                continue
            
            x, y, w, h = bbox
            image_crop = frame[int(y):int(y+h), int(x):int(x+w)]
            
            feature = self.extract_feature(image_crop)
            
            if feature is None:
                if track_id not in self.feature_extraction_failures:
                    self.feature_extraction_failures[track_id] = 0
                self.feature_extraction_failures[track_id] += 1
                
                if track_id in self.last_successful_feature:
                    feature = self.last_successful_feature[track_id]
                else:
                    continue
            else:
                self.feature_extraction_failures[track_id] = 0
                self.last_successful_feature[track_id] = feature
            
            color_signature = self._extract_color_histogram(image_crop)
            
            persistent_id = None
            match_confidence = 0.0
            match_method = "new"
            match_details = {}
            
            best_match_pid = None
            best_match_score = 0.0
            best_match_details = {}
            
            for pid, pid_features in self.feature_history.items():
                feature_similarities = [
                    self._calculate_similarity(feature, pid_feat) 
                    for pid_feat in pid_features
                ]
                avg_feature_sim = np.mean(feature_similarities)
                max_feature_sim = max(feature_similarities)
                
                color_sim = 0.0
                if color_signature and pid in self.color_histograms:
                    color_sim = self._calculate_color_similarity(
                        color_signature, 
                        self.color_histograms[pid]
                    )
                
                if pid in self.color_histograms and color_signature:
                    combined_score = (1.0 - self.color_weight) * avg_feature_sim + \
                                   self.color_weight * color_sim
                else:
                    combined_score = avg_feature_sim
                
                if pid in self.camera_locations and \
                   self.camera_locations[pid] == camera_id and \
                   pid in self.spatial_context:
                    
                    last_bbox = self.spatial_context[pid]['bbox']
                    last_time = self.spatial_context[pid]['timestamp']
                    time_diff = current_time - last_time
                    
                    if time_diff < self.spatial_time_window:
                        spatial_distance = self._calculate_spatial_distance(bbox, last_bbox)
                        
                        if spatial_distance < self.spatial_proximity_threshold:
                            spatial_bonus = self.spatial_proximity_bonus * \
                                          (1.0 - spatial_distance / self.spatial_proximity_threshold)
                            combined_score += spatial_bonus
                
                has_conflict, conflict_reason, conflict_penalty = \
                    self._check_assignment_conflicts(camera_id, pid, bbox, feature)
                
                if has_conflict:
                    combined_score *= (1.0 - conflict_penalty)
                    print(f"[ReID] ⚠️ Assignment conflict: {conflict_reason}")
                
                combined_score = self._apply_crossing_penalties(
                    camera_id, pid, combined_score, bbox
                )
                
                if combined_score > best_match_score:
                    best_match_score = combined_score
                    best_match_pid = pid
                    best_match_details = {
                        'feature_sim': avg_feature_sim,
                        'max_feature_sim': max_feature_sim,
                        'color_sim': color_sim,
                        'combined_score': combined_score,
                        'has_conflict': has_conflict
                    }
            
            threshold = self.cross_camera_threshold if \
                       best_match_pid and \
                       self.camera_locations.get(best_match_pid) != camera_id else \
                       self.similarity_threshold
            
            is_duplicate, duplicate_reason = self._check_duplicate_assignment(
                camera_id, best_match_pid, feature, bbox
            )
            
            if is_duplicate:
                print(f"[ReID] 🚫 Duplicate assignment blocked: {duplicate_reason}")
                best_match_score = 0.0
                best_match_pid = None
            
            if best_match_score >= threshold and best_match_pid is not None:
                persistent_id = best_match_pid
                match_confidence = best_match_score
                match_method = "matched"
                match_details = best_match_details
                
                self._add_feature_to_history(persistent_id, feature)
                self._update_color_histogram(persistent_id, color_signature)
                
            else:
                persistent_id = self.next_persistent_id
                self.next_persistent_id += 1
                match_confidence = 1.0
                match_method = "new"
                
                self.feature_history[persistent_id] = [self._normalize_feature(feature)]
                self.persistent_ids[persistent_id] = self._normalize_feature(feature)
                
                if color_signature:
                    self.color_histograms[persistent_id] = color_signature
                    self.color_confidence[persistent_id] = 1.0
                
                print(f"[ReID] 🆕 New UID: {persistent_id} (camera: {camera_id})")
            
            self.last_seen[persistent_id] = current_time
            self.camera_locations[persistent_id] = camera_id
            
            self.spatial_context[persistent_id] = {
                'bbox': bbox,
                'timestamp': current_time,
                'feature': feature
            }
            
            self._update_motion_history(persistent_id, bbox, current_time)
            self._record_assignment(camera_id, persistent_id, feature, bbox, current_time)
            
            if persistent_id not in self.camera_history:
                self.camera_history[persistent_id] = []
            if not self.camera_history[persistent_id] or \
               self.camera_history[persistent_id][-1][0] != camera_id:
                self.camera_history[persistent_id].append((camera_id, current_time))
            
            if persistent_id not in self.first_seen:
                self.first_seen[persistent_id] = current_time
            
            frame_state[persistent_id] = (bbox, feature)
            
            if self.db and match_method == "new":
                self._save_feature_to_db(persistent_id, camera_id, feature, match_confidence)
            
            result = {
                'detection': detection,
                'persistent_id': persistent_id,
                'track_id': track_id,
                'confidence': match_confidence,
                'method': match_method,
                'bbox': bbox,
                'camera_id': camera_id,
                'timestamp': current_time
            }
            
            if match_details:
                result['match_details'] = match_details
            
            results.append(result)
        
        if frame_state:
            crossing_pairs = self._detect_crossing_batch(camera_id, frame_state)
            if crossing_pairs:
                for result in results:
                    pid = result['persistent_id']
                    if self._is_crossing_active(camera_id, pid):
                        result['crossing_active'] = True
        
        active_pids = [r['persistent_id'] for r in results]
        self.active_uids_per_camera[camera_id] = set(active_pids)
        
        return results

    def get_person_info(self, persistent_id):
        if persistent_id not in self.persistent_ids:
            return None
        
        info = {
            'persistent_id': persistent_id,
            'first_seen': self.first_seen.get(persistent_id),
            'last_seen': self.last_seen.get(persistent_id),
            'current_camera': self.camera_locations.get(persistent_id),
            'camera_history': self.camera_history.get(persistent_id, []),
            'feature_count': len(self.feature_history.get(persistent_id, [])),
            'has_color_profile': persistent_id in self.color_histograms,
            'color_confidence': self.color_confidence.get(persistent_id, 0.0)
        }
        
        return info

    def cleanup_old_tracks(self, max_age=300.0):
        current_time = time.time()
        pids_to_remove = []
        
        for pid, last_time in self.last_seen.items():
            if current_time - last_time > max_age:
                pids_to_remove.append(pid)
        
        for pid in pids_to_remove:
            self.feature_history.pop(pid, None)
            self.persistent_ids.pop(pid, None)
            self.last_seen.pop(pid, None)
            self.camera_locations.pop(pid, None)
            self.camera_history.pop(pid, None)
            self.spatial_context.pop(pid, None)
            self.first_seen.pop(pid, None)
            self.color_histograms.pop(pid, None)
            self.color_confidence.pop(pid, None)
            self.position_history.pop(pid, None)
            self.velocity_estimates.pop(pid, None)
            
            print(f"[ReID] 🗑️ Cleaned up old track: PID {pid}")
        
        return len(pids_to_remove)

    def get_statistics(self):
        stats = {
            'total_persons': len(self.persistent_ids),
            'next_id': self.next_persistent_id,
            'active_cameras': len(self.active_uids_per_camera),
            'active_crossings': sum(len(pairs) for pairs in self.active_crossings.values()),
            'persons_with_color': len(self.color_histograms),
            'avg_features_per_person': np.mean([len(f) for f in self.feature_history.values()]) 
                                       if self.feature_history else 0,
            'persons_with_motion': len(self.velocity_estimates)
        }
        
        return stats

    def reset(self):
        self.persistent_ids.clear()
        self.feature_history.clear()
        self.track_to_persistent.clear()
        self.last_seen.clear()
        self.camera_locations.clear()
        self.camera_history.clear()
        self.spatial_context.clear()
        self.first_seen.clear()
        self.track_history.clear()
        self.recent_candidates.clear()
        self.pending_cross_matches.clear()
        self.active_uids_per_camera.clear()
        self.uid_assignment_lock.clear()
        self.feature_extraction_failures.clear()
        self.last_successful_feature.clear()
        self.recent_assignments.clear()
        self.position_history.clear()
        self.velocity_estimates.clear()
        self.active_crossings.clear()
        self.assignment_locks.clear()
        self.last_assignments.clear()
        self.color_histograms.clear()
        self.color_confidence.clear()
        
        print("[ReID] ♻️ Tracker reset complete")
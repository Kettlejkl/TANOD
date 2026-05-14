"""
Tracking Manager Module
Orchestrates all tracking components with improved matching logic
FIXES: Direct bbox matching, track recovery, better temporal handling
"""

import time
import logging
from threading import Lock
import numpy as np


class TrackingManager:
    """Main orchestrator for person tracking system"""
    
    def __init__(self,
                 feature_extractor,
                 appearance_analyzer,
                 motion_tracker,
                 feature_matcher,
                 person_database,
                 min_box_area=2500,
                 log_level=logging.INFO):
        
        self.logger = logging.getLogger('TrackingManager')
        self.logger.setLevel(log_level)
        
        self.feature_extractor = feature_extractor
        self.appearance_analyzer = appearance_analyzer
        self.motion_tracker = motion_tracker
        self.feature_matcher = feature_matcher
        self.person_database = person_database
        
        self.min_box_area = min_box_area
        
        self._lock = Lock()
        
        self.metrics = {
            'total_detections': 0,
            'new_persons': 0,
            'matches': 0,
            'recoveries': 0,  # NEW: Track recoveries
            'avg_processing_time': 0.0,
            'false_negatives': 0,
            'ambiguous_matches': 0
        }
        
        self.frame_count = 0
        self.last_frame_person_positions = {}
        self.person_continuity_scores = {}
        
        # NEW: Track recently lost persons for recovery
        self.recently_lost_persons = {}  # person_id -> {bbox, timestamp, feature}
        self.lost_person_timeout = 10.0  # Keep for 10 seconds
        
        # NEW: Track bbox-to-person mapping without relying on track_id
        self.bbox_history = {}  # person_id -> list of recent bboxes
        
        self.logger.info("Tracking manager initialized with track recovery")
    
    def _is_valid_detection(self, bbox):
        """Check if detection bbox is valid"""
        if bbox is None or len(bbox) != 4:
            return False
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return area >= self.min_box_area
    
    def _filter_overlapping_detections(self, detections):
        """Remove duplicate detections with high IoU"""
        if len(detections) <= 1:
            return detections
        
        filtered = []
        
        for det1 in detections:
            is_duplicate = False
            for det2 in filtered:
                iou = self.feature_matcher.calculate_bbox_iou(
                    det1['bbox'], det2['bbox']
                )
                if iou > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(det1)
        
        return filtered
    
    def _calculate_bbox_size_similarity(self, bbox1, bbox2):
        """Calculate size similarity between two bboxes"""
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        
        area1 = w1 * h1
        area2 = w2 * h2
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        size_ratio = min(area1, area2) / max(area1, area2)
        aspect_ratio1 = w1 / h1 if h1 > 0 else 1
        aspect_ratio2 = w2 / h2 if h2 > 0 else 1
        aspect_similarity = min(aspect_ratio1, aspect_ratio2) / max(aspect_ratio1, aspect_ratio2)
        
        return 0.7 * size_ratio + 0.3 * aspect_similarity
    
    def _get_temporal_consistency_bonus(self, person_id, current_bbox, timestamp):
        """Calculate bonus based on temporal consistency - IMPROVED"""
        if person_id not in self.last_frame_person_positions:
            # NEW: Check recently lost persons
            if person_id in self.recently_lost_persons:
                lost_info = self.recently_lost_persons[person_id]
                time_delta = timestamp - lost_info['timestamp']
                
                if time_delta <= self.lost_person_timeout:
                    distance = self.feature_matcher.calculate_spatial_distance(
                        current_bbox, lost_info['bbox']
                    )
                    
                    # More lenient for recovered persons
                    max_expected_distance = 300 * time_delta
                    
                    if distance < max_expected_distance:
                        consistency = 1.0 - (distance / max_expected_distance)
                        return 0.20 * consistency  # Good recovery bonus
            
            return 0.0
        
        last_position = self.last_frame_person_positions[person_id]
        time_delta = timestamp - last_position['timestamp']
        
        # CHANGED: Extended from 3.0 to 10.0 seconds
        if time_delta > 10.0:
            return 0.0
        
        distance = self.feature_matcher.calculate_spatial_distance(
            current_bbox, last_position['bbox']
        )
        
        # CHANGED: More lenient distance expectations
        max_expected_distance = 250 * time_delta  # Increased from 200
        
        if distance < max_expected_distance:
            consistency = 1.0 - (distance / max_expected_distance)
            return 0.25 * consistency
        
        return 0.0
    
    def _update_continuity_score(self, person_id, matched):
        """Update continuity score for person"""
        if person_id not in self.person_continuity_scores:
            self.person_continuity_scores[person_id] = 0.0
        
        if matched:
            self.person_continuity_scores[person_id] = min(1.0, self.person_continuity_scores[person_id] + 0.15)
        else:
            # CHANGED: Less aggressive penalty
            self.person_continuity_scores[person_id] = max(0.0, self.person_continuity_scores[person_id] - 0.1)
    
    def _get_continuity_bonus(self, person_id):
        """Get bonus based on tracking continuity"""
        score = self.person_continuity_scores.get(person_id, 0.0)
        return 0.20 * score
    
    # NEW: Track recovery mechanism
    def _try_recover_lost_person(self, bbox, feature, camera_id, timestamp):
        """Attempt to match detection with recently lost persons"""
        best_match_id = None
        best_match_score = 0.0
        
        for person_id, lost_info in list(self.recently_lost_persons.items()):
            # Check if too old
            if timestamp - lost_info['timestamp'] > self.lost_person_timeout:
                del self.recently_lost_persons[person_id]
                continue
            
            # Check camera match
            person_data = self.person_database.get_person(person_id)
            if person_data is None:
                del self.recently_lost_persons[person_id]
                continue
            
            # Feature similarity
            feature_sims = self.feature_matcher.calculate_similarity_batch(
                feature, person_data['features']
            )
            
            if not feature_sims:
                continue
            
            avg_feature_sim = max(feature_sims)
            
            # Spatial proximity
            spatial_distance = self.feature_matcher.calculate_spatial_distance(
                bbox, lost_info['bbox']
            )
            
            time_delta = timestamp - lost_info['timestamp']
            max_movement = 300 * time_delta
            
            if spatial_distance > max_movement:
                continue
            
            spatial_score = 1.0 - (spatial_distance / max_movement)
            
            # Combined score
            combined_score = 0.7 * avg_feature_sim + 0.3 * spatial_score
            
            if combined_score > best_match_score:
                best_match_score = combined_score
                best_match_id = person_id
        
        # CHANGED: More lenient recovery threshold
        if best_match_score > 0.40:  # Lower than normal matching
            return best_match_id, best_match_score
        
        return None, 0.0
    
    # NEW: Update recently lost persons
    def _update_lost_persons(self, current_person_ids, timestamp):
        """Track persons that disappeared this frame"""
        all_known_persons = set(self.last_frame_person_positions.keys())
        disappeared = all_known_persons - current_person_ids
        
        for person_id in disappeared:
            if person_id not in self.recently_lost_persons:
                last_pos = self.last_frame_person_positions[person_id]
                person_data = self.person_database.get_person(person_id)
                
                if person_data is not None and person_data['current_feature'] is not None:
                    self.recently_lost_persons[person_id] = {
                        'bbox': last_pos['bbox'],
                        'timestamp': last_pos['timestamp'],
                        'feature': person_data['current_feature']
                    }
        
        # Clean up old lost persons
        for person_id in list(self.recently_lost_persons.keys()):
            if timestamp - self.recently_lost_persons[person_id]['timestamp'] > self.lost_person_timeout:
                del self.recently_lost_persons[person_id]
    
    def _perform_bipartite_matching(self, detections, candidates, camera_id, frame, timestamp):
        """Perform optimal bipartite matching between detections and persons"""
        if not detections or not candidates:
            return {}
        
        cost_matrix = np.zeros((len(detections), len(candidates)))
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            
            x, y, w, h = bbox
            image_crop = frame[int(y):int(y+h), int(x):int(x+w)]
            feature = self.feature_extractor.extract(image_crop)
            
            if feature is None:
                cost_matrix[i, :] = 1.0
                continue
            
            color = self.appearance_analyzer.extract_color_histogram(image_crop)
            clothing = self.appearance_analyzer.extract_clothing_attributes(image_crop)
            
            for j, (person_id, person_data) in enumerate(candidates):
                feature_sims = self.feature_matcher.calculate_similarity_batch(
                    feature, person_data['features']
                )
                
                if not feature_sims:
                    cost_matrix[i, j] = 1.0
                    continue
                
                avg_feature_sim = max(feature_sims)
                
                color_sim = 0.0
                if color and person_data['color']:
                    color_sim = self.appearance_analyzer.compare_color_histograms(
                        color, person_data['color']
                    )
                
                clothing_sim = 0.0
                if clothing and person_data['clothing']:
                    clothing_sim = self.appearance_analyzer.compare_clothing_attributes(
                        clothing, person_data['clothing']
                    )
                
                motion_score = self.motion_tracker.calculate_motion_consistency(
                    person_id, bbox, timestamp
                )
                
                spatial_bonus = 0.0
                distance = float('inf')
                
                # CHANGED: More lenient spatial matching
                if person_data['spatial_context']:
                    distance = self.feature_matcher.calculate_spatial_distance(
                        bbox, person_data['spatial_context']['bbox']
                    )
                    
                    # CHANGED: Increased from 800 to 1200
                    if distance < 1200:
                        spatial_bonus = self.feature_matcher.calculate_spatial_bonus(
                            bbox,
                            person_data['spatial_context']['bbox'],
                            person_data['spatial_context']['timestamp'],
                            timestamp
                        )
                
                size_similarity = 0.0
                if person_data['spatial_context']:
                    size_similarity = self._calculate_bbox_size_similarity(
                        bbox, person_data['spatial_context']['bbox']
                    )
                
                temporal_bonus = self._get_temporal_consistency_bonus(person_id, bbox, timestamp)
                continuity_bonus = self._get_continuity_bonus(person_id)
                
                combined_score = self.feature_matcher.compute_combined_score(
                    avg_feature_sim,
                    color_sim,
                    clothing_sim,
                    motion_score,
                    spatial_bonus
                )
                
                combined_score += temporal_bonus
                combined_score += continuity_bonus
                combined_score += 0.10 * size_similarity
                
                cost_matrix[i, j] = 1.0 - combined_score
        
        from scipy.optimize import linear_sum_assignment
        det_indices, person_indices = linear_sum_assignment(cost_matrix)
        
        matches = {}
        for det_idx, person_idx in zip(det_indices, person_indices):
            score = 1.0 - cost_matrix[det_idx, person_idx]
            person_id = candidates[person_idx][0]
            
            same_camera = candidates[person_idx][1]['current_camera'] == camera_id
            threshold = self.feature_matcher.get_threshold(same_camera)
            
            if score >= threshold:
                matches[det_idx] = (person_id, score)
        
        return matches
    
    def process_detections(self, camera_id, detections, frame, timestamp=None):
        """Process detections from a camera"""
        with self._lock:
            return self._process_detections_internal(camera_id, detections, frame, timestamp)
    
    def _process_detections_internal(self, camera_id, detections, frame, timestamp=None):
        """Internal detection processing with bipartite matching and recovery"""
        if timestamp is None:
            timestamp = time.time()
        
        detections = self._filter_overlapping_detections(detections)
        
        start_time = time.time()
        results = []
        
        self.frame_count += 1
        
        all_persons = self.person_database.get_all_persons()
        candidates = [(pid, self.person_database.get_person(pid)) for pid in all_persons]
        
        matches = self._perform_bipartite_matching(detections, candidates, camera_id, frame, timestamp)
        
        matched_person_ids = set()
        
        for det_idx, detection in enumerate(detections):
            self.metrics['total_detections'] += 1
            
            bbox = detection.get('bbox')
            track_id = detection.get('track_id')  # May be None or unstable
            
            if not self._is_valid_detection(bbox):
                continue
            
            x, y, w, h = bbox
            image_crop = frame[int(y):int(y+h), int(x):int(x+w)]
            
            feature = self.feature_extractor.extract(image_crop)
            if feature is None:
                continue
            
            color = self.appearance_analyzer.extract_color_histogram(image_crop)
            clothing = self.appearance_analyzer.extract_clothing_attributes(image_crop)
            
            person_id = None
            best_match_score = 0.0
            match_method = "new"
            
            # Try bipartite matching first
            if det_idx in matches:
                person_id, best_match_score = matches[det_idx]
                match_method = "matched"
                self.metrics['matches'] += 1
                matched_person_ids.add(person_id)
                
                self.person_database.update_person(
                    person_id,
                    feature=feature,
                    camera_id=camera_id,
                    timestamp=timestamp,
                    bbox=bbox,
                    color=color,
                    clothing=clothing
                )
                
                self._update_continuity_score(person_id, True)
                
                # Remove from lost persons if recovered
                if person_id in self.recently_lost_persons:
                    del self.recently_lost_persons[person_id]
                
                if self.frame_count % 30 == 0:
                    print(f"✅ MATCHED to ID: {person_id} (score: {best_match_score:.3f})")
            
            # NEW: Try recovery from recently lost persons
            elif self.recently_lost_persons:
                recovered_id, recovery_score = self._try_recover_lost_person(
                    bbox, feature, camera_id, timestamp
                )
                
                if recovered_id is not None:
                    person_id = recovered_id
                    best_match_score = recovery_score
                    match_method = "recovered"
                    self.metrics['recoveries'] += 1
                    matched_person_ids.add(person_id)
                    
                    self.person_database.update_person(
                        person_id,
                        feature=feature,
                        camera_id=camera_id,
                        timestamp=timestamp,
                        bbox=bbox,
                        color=color,
                        clothing=clothing
                    )
                    
                    self._update_continuity_score(person_id, True)
                    
                    del self.recently_lost_persons[person_id]
                    
                    if self.frame_count % 30 == 0:
                        print(f"🔄 RECOVERED ID: {person_id} (score: {recovery_score:.3f})")
            
            # Create new person if no match found
            if person_id is None:
                person_id = self.person_database.create_person(
                    feature, camera_id, timestamp, color, clothing
                )
                match_method = "new"
                best_match_score = 0.0
                self.metrics['new_persons'] += 1
                matched_person_ids.add(person_id)
                
                self._update_continuity_score(person_id, False)
                
                if self.frame_count % 30 == 0:
                    print(f"⭐ NEW ID: {person_id}")
            
            self.last_frame_person_positions[person_id] = {
                'bbox': bbox,
                'timestamp': timestamp
            }
            
            smoothed_bbox = self.motion_tracker.update(person_id, bbox, timestamp)
            
            result = {
                'detection': detection,
                'persistent_id': person_id,
                'track_id': track_id,  # Keep original track_id for reference
                'confidence': best_match_score,
                'method': match_method,
                'bbox': bbox,
                'smoothed_bbox': smoothed_bbox,
                'camera_id': camera_id,
                'timestamp': timestamp
            }
            
            results.append(result)
        
        # NEW: Update recently lost persons
        self._update_lost_persons(matched_person_ids, timestamp)
        
        # Clean up old positions
        old_positions = list(self.last_frame_person_positions.keys())
        current_persons = [r['persistent_id'] for r in results]
        for pid in old_positions:
            if pid not in current_persons:
                time_since = timestamp - self.last_frame_person_positions[pid]['timestamp']
                # CHANGED: Extended from 3.0 to 10.0
                if time_since > 10.0:
                    del self.last_frame_person_positions[pid]
        
        self.person_database.cleanup_old_persons(timestamp)
        self.person_database.enforce_memory_limits(timestamp)
        
        processing_time = time.time() - start_time
        alpha = 0.1
        self.metrics['avg_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.metrics['avg_processing_time']
        )
        
        return results
    
    def cleanup_old_tracks(self, max_age=30.0):
        """Remove old tracks that haven't been updated recently"""
        current_time = time.time()
        removed = self.person_database.cleanup_old_persons(current_time)
        if removed:
            for person_id in removed:
                self.motion_tracker.remove_person(person_id)
                self.last_frame_person_positions.pop(person_id, None)
                self.person_continuity_scores.pop(person_id, None)
                self.recently_lost_persons.pop(person_id, None)  # NEW
    
    def get_person_info(self, person_id):
        """Get information about a person"""
        person_data = self.person_database.get_person(person_id)
        if person_data is None:
            return None
        
        info = {
            'persistent_id': person_id,
            'first_seen': person_data['first_seen'],
            'last_seen': person_data['last_seen'],
            'current_camera': person_data['current_camera'],
            'camera_history': person_data['camera_history'],
            'feature_count': len(person_data['features']),
            'has_color_profile': person_data['color'] is not None,
            'color_confidence': person_data['color_confidence'],
            'has_clothing_profile': person_data['clothing'] is not None,
            'trajectory': self.motion_tracker.get_trajectory(person_id),
            'continuity_score': self.person_continuity_scores.get(person_id, 0.0),
            'is_recently_lost': person_id in self.recently_lost_persons  # NEW
        }
        
        return info
    
    def get_statistics(self):
        """Get tracking statistics"""
        db_stats = self.person_database.get_statistics()
        
        match_rate = 0.0
        if self.metrics['total_detections'] > 0:
            match_rate = self.metrics['matches'] / self.metrics['total_detections']
        
        recovery_rate = 0.0
        if self.metrics['total_detections'] > 0:
            recovery_rate = self.metrics['recoveries'] / self.metrics['total_detections']
        
        return {
            **db_stats,
            'metrics': self.metrics.copy(),
            'match_rate': match_rate,
            'recovery_rate': recovery_rate,  # NEW
            'recently_lost_count': len(self.recently_lost_persons),  # NEW
            'avg_continuity': (
                sum(self.person_continuity_scores.values()) / len(self.person_continuity_scores)
                if self.person_continuity_scores else 0.0
            )
        }
    
    def reset(self):
        """Reset all tracking state"""
        with self._lock:
            self.person_database.clear()
            self.motion_tracker.clear()
            
            for key in self.metrics:
                self.metrics[key] = 0 if isinstance(self.metrics[key], int) else 0.0
            
            self.frame_count = 0
            self.last_frame_person_positions.clear()
            self.person_continuity_scores.clear()
            self.recently_lost_persons.clear()  # NEW
            self.bbox_history.clear()  # NEW
            
            self.logger.info("Tracking manager reset complete")
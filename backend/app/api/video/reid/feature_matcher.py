"""
Feature Matching Module
Handles similarity computation and matching logic
FIXES: Better thresholds, improved spatial/temporal handling
"""

import numpy as np
from scipy.spatial.distance import cosine, cdist
from collections import deque
import logging


class FeatureMatcher:
    """Matches features between detections and known persons"""
    
    def __init__(self,
                 similarity_threshold=0.50,  # CHANGED: from 0.15 to 0.50
                 cross_camera_threshold=0.40,  # CHANGED: from 0.25 to 0.40
                 spatial_proximity_threshold=400,  # CHANGED: from 200 to 400
                 spatial_proximity_bonus=0.30,
                 spatial_time_window=30.0,  # CHANGED: from 15.0 to 30.0
                 color_weight=0.25,
                 clothing_weight=0.25,
                 velocity_weight=0.20,
                 iou_weight=0.25,
                 continuity_bonus=0.30,
                 min_feature_separation=0.08,  # CHANGED: from 0.12 to 0.08
                 log_level=logging.INFO):
        
        self.logger = logging.getLogger('FeatureMatcher')
        self.logger.setLevel(log_level)
        
        self.similarity_threshold = similarity_threshold
        self.cross_camera_threshold = cross_camera_threshold
        
        self.spatial_proximity_threshold = spatial_proximity_threshold
        self.spatial_proximity_bonus = spatial_proximity_bonus
        self.spatial_time_window = spatial_time_window
        
        self.color_weight = color_weight
        self.clothing_weight = clothing_weight
        self.velocity_weight = velocity_weight
        self.iou_weight = iou_weight
        self.continuity_bonus = continuity_bonus
        
        self.min_feature_separation = min_feature_separation
        
        self.logger.info(f"FeatureMatcher initialized - threshold: {similarity_threshold:.2f}")
    
    def calculate_similarity(self, feature1, feature2):
        """Calculate cosine similarity between two features"""
        if feature1 is None or feature2 is None:
            return 0.0
        return 1 - cosine(feature1, feature2)
    
    def calculate_similarity_batch(self, query_feature, gallery_features):
        """Calculate similarities between query and multiple gallery features"""
        if not gallery_features:
            return []
        
        try:
            gallery_array = np.array(gallery_features)
            similarities = 1 - cdist([query_feature], gallery_array, metric='cosine')[0]
            return similarities.tolist()
        except:
            return [self.calculate_similarity(query_feature, f) for f in gallery_features]
    
    def check_feature_separation(self, feature1, feature2):
        """Check if two features are sufficiently different"""
        similarity = self.calculate_similarity(feature1, feature2)
        return (1.0 - similarity) >= self.min_feature_separation
    
    def calculate_spatial_distance(self, bbox1, bbox2):
        """Calculate Euclidean distance between bbox centers"""
        if bbox1 is None or bbox2 is None:
            return float('inf')
        
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_bbox_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union for two bboxes"""
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        if len(bbox1) == 4 and len(bbox2) == 4:
            if bbox1[2] < bbox1[0] or bbox1[3] < bbox1[1]:
                bbox1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
            if bbox2[2] < bbox2[0] or bbox2[3] < bbox2[1]:
                bbox2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]
        
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
    
    def compute_combined_score(self,
                               feature_sim,
                               color_sim=None,
                               clothing_sim=None,
                               motion_score=None,
                               spatial_bonus=0.0,
                               iou_boost=0.0,
                               continuity_bonus=0.0):
        """Compute combined matching score from multiple cues"""
        
        score = feature_sim
        
        if color_sim is not None and color_sim > 0:
            score = (1.0 - self.color_weight) * score + self.color_weight * color_sim
        
        if clothing_sim is not None and clothing_sim > 0:
            score = (1.0 - self.clothing_weight) * score + self.clothing_weight * clothing_sim
        
        if motion_score is not None:
            score += self.velocity_weight * motion_score
        
        score += spatial_bonus
        score += iou_boost
        score += continuity_bonus
        
        return score
    
    def calculate_spatial_bonus(self, bbox, last_bbox, last_time, timestamp):
        """Calculate spatial proximity bonus - IMPROVED"""
        time_diff = timestamp - last_time
        
        if time_diff >= self.spatial_time_window:
            return 0.0
        
        spatial_distance = self.calculate_spatial_distance(bbox, last_bbox)
        
        # CHANGED: More gradual falloff
        if spatial_distance < self.spatial_proximity_threshold:
            distance_factor = 1.0 - (spatial_distance / self.spatial_proximity_threshold)
            
            # CHANGED: Time-based scaling - more lenient for longer gaps
            time_factor = 1.0 - (time_diff / self.spatial_time_window)
            
            bonus = self.spatial_proximity_bonus * distance_factor * time_factor
            return bonus
        
        # CHANGED: Small bonus even beyond threshold for nearby detections
        elif spatial_distance < self.spatial_proximity_threshold * 2:
            extended_distance_factor = 1.0 - ((spatial_distance - self.spatial_proximity_threshold) / self.spatial_proximity_threshold)
            return self.spatial_proximity_bonus * 0.3 * extended_distance_factor
        
        return 0.0
    
    def get_threshold(self, same_camera):
        """Get appropriate similarity threshold based on context"""
        return self.similarity_threshold if same_camera else self.cross_camera_threshold
    
    def calculate_trajectory_consistency(self, trajectory, new_bbox):
        """
        NEW: Calculate how consistent new detection is with trajectory
        """
        if not trajectory or len(trajectory) < 2:
            return 0.5
        
        recent = trajectory[-5:]  # Last 5 positions
        
        # Calculate average velocity from trajectory
        velocities = []
        for i in range(1, len(recent)):
            dt = recent[i]['timestamp'] - recent[i-1]['timestamp']
            if dt > 0:
                dx = recent[i]['position'][0] - recent[i-1]['position'][0]
                dy = recent[i]['position'][1] - recent[i-1]['position'][1]
                velocities.append((dx/dt, dy/dt))
        
        if not velocities:
            return 0.5
        
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        # Predict next position
        last_pos = recent[-1]['position']
        time_delta = 1.0  # Assume 1 second
        predicted_x = last_pos[0] + avg_vx * time_delta
        predicted_y = last_pos[1] + avg_vy * time_delta
        
        # Compare with actual
        actual_x = (new_bbox[0] + new_bbox[2]) / 2
        actual_y = (new_bbox[1] + new_bbox[3]) / 2
        
        distance = np.sqrt((actual_x - predicted_x)**2 + (actual_y - predicted_y)**2)
        
        max_deviation = 500  # pixels
        consistency = max(0.0, 1.0 - (distance / max_deviation))
        
        return consistency
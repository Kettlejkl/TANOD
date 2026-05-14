"""
Person Database Module
Manages person identities and their associated data
FIXES: Longer retention, better memory management
"""

import time
from collections import deque
import logging


class PersonDatabase:
    """Central database for person identities and metadata"""
    
    def __init__(self,
                 max_features_per_person=300,
                 max_tracked_persons=2000,
                 max_age=3600.0,  # CHANGED: from 1800.0 to 3600.0 (1 hour)
                 log_level=logging.INFO):
        
        self.logger = logging.getLogger('PersonDatabase')
        self.logger.setLevel(log_level)
        
        self.max_features_per_person = max_features_per_person
        self.max_tracked_persons = max_tracked_persons
        self.max_age = max_age
        
        self.next_id = 1
        self.feature_history = {}
        self.current_features = {}
        
        self.first_seen = {}
        self.last_seen = {}
        self.camera_locations = {}
        self.camera_history = {}
        self.spatial_context = {}
        
        self.color_histograms = {}
        self.color_confidence = {}
        self.clothing_attributes = {}
        
        # NEW: Track update frequency for smarter cleanup
        self.update_counts = {}
        
        self.logger.info(f"PersonDatabase initialized (max_age={max_age:.0f}s)")
    
    def create_person(self, feature, camera_id, timestamp, color=None, clothing=None):
        """Create a new person identity"""
        person_id = self.next_id
        self.next_id += 1
        
        self.feature_history[person_id] = [feature]
        self.current_features[person_id] = feature
        
        self.first_seen[person_id] = timestamp
        self.last_seen[person_id] = timestamp
        self.camera_locations[person_id] = camera_id
        self.camera_history[person_id] = [(camera_id, timestamp)]
        
        self.update_counts[person_id] = 1  # NEW
        
        if color:
            self.color_histograms[person_id] = color
            self.color_confidence[person_id] = 1.0
        
        if clothing:
            self.clothing_attributes[person_id] = clothing
        
        self.logger.info(f"Created new person ID: {person_id}")
        return person_id
    
    def update_person(self, person_id, feature=None, camera_id=None, 
                     timestamp=None, bbox=None, color=None, clothing=None):
        """Update person data"""
        if person_id not in self.feature_history:
            return False
        
        # NEW: Increment update count
        self.update_counts[person_id] = self.update_counts.get(person_id, 0) + 1
        
        if feature is not None:
            if len(self.feature_history[person_id]) >= self.max_features_per_person:
                # CHANGED: Remove oldest feature more intelligently
                # Keep features from different time periods
                self.feature_history[person_id].pop(0)
            self.feature_history[person_id].append(feature)
            self.current_features[person_id] = feature
        
        if timestamp is not None:
            self.last_seen[person_id] = timestamp
        
        if camera_id is not None:
            self.camera_locations[person_id] = camera_id
            
            if (not self.camera_history[person_id] or 
                self.camera_history[person_id][-1][0] != camera_id):
                self.camera_history[person_id].append((camera_id, timestamp))
        
        if bbox is not None and timestamp is not None:
            self.spatial_context[person_id] = {
                'bbox': bbox,
                'timestamp': timestamp,
                'camera': camera_id
            }
        
        if color is not None:
            self._update_color(person_id, color)
        
        if clothing is not None:
            self.clothing_attributes[person_id] = clothing
        
        return True
    
    def _update_color(self, person_id, new_color):
        """Update color histogram with exponential moving average"""
        if person_id not in self.color_histograms:
            self.color_histograms[person_id] = new_color
            self.color_confidence[person_id] = 1.0
        else:
            old_color = self.color_histograms[person_id]
            for key in ['upper_hue', 'upper_sat', 'lower_hue', 'lower_sat']:
                if key in old_color and key in new_color:
                    # CHANGED: More aggressive update for recent colors
                    old_color[key] = 0.6 * old_color[key] + 0.4 * new_color[key]  # from 0.7/0.3
            self.color_histograms[person_id] = old_color
            self.color_confidence[person_id] = min(1.0, self.color_confidence[person_id] + 0.1)
    
    def get_person(self, person_id):
        """Get complete person data"""
        if person_id not in self.feature_history:
            return None
        
        return {
            'id': person_id,
            'features': self.feature_history[person_id],
            'current_feature': self.current_features.get(person_id),
            'first_seen': self.first_seen.get(person_id),
            'last_seen': self.last_seen.get(person_id),
            'current_camera': self.camera_locations.get(person_id),
            'camera_history': self.camera_history.get(person_id, []),
            'spatial_context': self.spatial_context.get(person_id),
            'color': self.color_histograms.get(person_id),
            'color_confidence': self.color_confidence.get(person_id, 0.0),
            'clothing': self.clothing_attributes.get(person_id),
            'update_count': self.update_counts.get(person_id, 0)  # NEW
        }
    
    def get_all_persons(self):
        """Get list of all person IDs"""
        return list(self.feature_history.keys())
    
    def get_features(self, person_id):
        """Get feature history for a person"""
        return self.feature_history.get(person_id, [])
    
    def remove_person(self, person_id):
        """Remove a person and all associated data"""
        for storage in [self.feature_history, self.current_features, self.first_seen,
                       self.last_seen, self.camera_locations, self.camera_history,
                       self.spatial_context, self.color_histograms, self.color_confidence,
                       self.clothing_attributes, self.update_counts]:  # NEW
            storage.pop(person_id, None)
        
        self.logger.debug(f"Removed person ID: {person_id}")
    
    def cleanup_old_persons(self, current_time=None):
        """Remove persons not seen recently - IMPROVED"""
        if current_time is None:
            current_time = time.time()
        
        cutoff_time = current_time - self.max_age
        removed = []
        
        for person_id, last_time in list(self.last_seen.items()):
            if last_time < cutoff_time:
                # NEW: Keep persons with high update counts longer
                update_count = self.update_counts.get(person_id, 0)
                
                # If person was seen many times, give them extra time
                if update_count > 50:
                    extended_cutoff = current_time - (self.max_age * 1.5)
                    if last_time > extended_cutoff:
                        continue  # Keep them longer
                
                self.remove_person(person_id)
                removed.append(person_id)
        
        if removed:
            self.logger.info(f"Cleaned up {len(removed)} old persons")
        
        return removed
    
    def enforce_memory_limits(self, current_time=None):
        """Enforce maximum tracked persons limit - IMPROVED"""
        if len(self.feature_history) <= self.max_tracked_persons:
            return []
        
        if current_time is None:
            current_time = time.time()
        
        # NEW: Score persons by (recency * update_count) to keep important ones
        person_scores = []
        for person_id in self.feature_history.keys():
            last_time = self.last_seen[person_id]
            update_count = self.update_counts.get(person_id, 1)
            
            time_since = current_time - last_time
            recency_score = 1.0 / (1.0 + time_since)  # Higher = more recent
            
            # Combined score: prefer recent and frequently updated persons
            score = recency_score * min(update_count / 100.0, 1.0)
            person_scores.append((person_id, last_time, score))
        
        # Sort by score (lowest first = least important)
        sorted_persons = sorted(person_scores, key=lambda x: x[2])
        
        num_to_remove = len(self.feature_history) - self.max_tracked_persons
        removed = []
        
        for person_id, last_time, score in sorted_persons[:num_to_remove]:
            # CHANGED: Only remove if not seen in last minute
            if current_time - last_time > 60:
                self.remove_person(person_id)
                removed.append(person_id)
        
        if removed:
            self.logger.info(f"Removed {len(removed)} persons to enforce memory limits")
        
        return removed
    
    def get_statistics(self):
        """Get database statistics"""
        avg_updates = 0.0
        if self.update_counts:
            avg_updates = sum(self.update_counts.values()) / len(self.update_counts)
        
        return {
            'total_persons': len(self.feature_history),
            'next_id': self.next_id,
            'persons_with_color': len(self.color_histograms),
            'persons_with_clothing': len(self.clothing_attributes),
            'avg_features_per_person': (
                sum(len(f) for f in self.feature_history.values()) / len(self.feature_history)
                if self.feature_history else 0
            ),
            'avg_updates_per_person': avg_updates  # NEW
        }
    
    def clear(self):
        """Clear all data"""
        self.next_id = 1
        self.feature_history.clear()
        self.current_features.clear()
        self.first_seen.clear()
        self.last_seen.clear()
        self.camera_locations.clear()
        self.camera_history.clear()
        self.spatial_context.clear()
        self.color_histograms.clear()
        self.color_confidence.clear()
        self.clothing_attributes.clear()
        self.update_counts.clear()  # NEW
        self.logger.info("Database cleared")
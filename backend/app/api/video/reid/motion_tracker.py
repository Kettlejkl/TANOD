"""
Motion Tracking Module
Handles Kalman filtering, velocity estimation, and trajectory tracking
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
import logging


@dataclass
class KalmanFilter:
    """Simple Kalman filter for position smoothing"""
    x: np.ndarray
    P: np.ndarray
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return self.x


class MotionTracker:
    """Tracks motion, velocity, and trajectories for persons"""
    
    def __init__(self, 
                 kalman_enabled=True,
                 max_history=20,
                 max_trajectory=60,
                 log_level=logging.INFO):
        
        self.logger = logging.getLogger('MotionTracker')
        self.logger.setLevel(log_level)
        
        self.kalman_enabled = kalman_enabled
        self.max_history = max_history
        self.max_trajectory = max_trajectory
        
        self.kalman_filters = {}
        self.position_history = {}
        self.velocity_estimates = {}
        self.trajectory_history = {}
    
    def _init_kalman(self, bbox):
        """Initialize Kalman filter for a new track"""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        x = np.array([[center_x], [center_y], [0], [0]])
        P = np.eye(4) * 1000
        F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        Q = np.eye(4) * 0.1
        R = np.eye(2) * 10
        
        return KalmanFilter(x, P, F, H, Q, R)
    
    def update(self, person_id, bbox, timestamp):
        """Update motion tracking for a person"""
        if bbox is None:
            return bbox
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        if person_id not in self.position_history:
            self.position_history[person_id] = deque(maxlen=self.max_history)
        self.position_history[person_id].append((center_x, center_y, timestamp))
        
        if len(self.position_history[person_id]) >= 2:
            recent = list(self.position_history[person_id])[-5:]
            if len(recent) >= 2:
                dt = recent[-1][2] - recent[0][2]
                if dt > 0:
                    dx = recent[-1][0] - recent[0][0]
                    dy = recent[-1][1] - recent[0][1]
                    self.velocity_estimates[person_id] = (dx/dt, dy/dt)
        
        if person_id not in self.trajectory_history:
            self.trajectory_history[person_id] = deque(maxlen=self.max_trajectory)
        self.trajectory_history[person_id].append({
            'position': (center_x, center_y),
            'timestamp': timestamp,
            'bbox': bbox
        })
        
        smoothed_bbox = self._apply_kalman(person_id, bbox)
        
        return smoothed_bbox
    
    def _apply_kalman(self, person_id, bbox):
        """Apply Kalman filter smoothing to bbox"""
        if not self.kalman_enabled:
            return bbox
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        if person_id not in self.kalman_filters:
            self.kalman_filters[person_id] = self._init_kalman(bbox)
        
        kf = self.kalman_filters[person_id]
        z = np.array([[center_x], [center_y]])
        
        kf.predict()
        kf.update(z)
        
        smoothed_x = kf.x[0, 0]
        smoothed_y = kf.x[1, 0]
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return (
            smoothed_x - width / 2,
            smoothed_y - height / 2,
            smoothed_x + width / 2,
            smoothed_y + height / 2
        )
    
    def calculate_motion_consistency(self, person_id, new_bbox, timestamp):
        """Calculate how consistent new position is with predicted motion"""
        if person_id not in self.position_history or not self.position_history[person_id]:
            return 0.5
        
        last_pos = self.position_history[person_id][-1]
        time_delta = timestamp - last_pos[2]
        
        if time_delta <= 0:
            return 0.5
        
        if person_id in self.velocity_estimates:
            vx, vy = self.velocity_estimates[person_id]
            predicted_x = last_pos[0] + vx * time_delta
            predicted_y = last_pos[1] + vy * time_delta
            
            actual_x = (new_bbox[0] + new_bbox[2]) / 2
            actual_y = (new_bbox[1] + new_bbox[3]) / 2
            
            distance = np.sqrt((actual_x - predicted_x)**2 + (actual_y - predicted_y)**2)
            max_expected_movement = 500 * time_delta
            
            if distance > max_expected_movement:
                return 0.0
            
            return max(0.0, 1.0 - (distance / max_expected_movement))
        
        return 0.5
    
    def get_velocity(self, person_id):
        """Get current velocity estimate for a person"""
        return self.velocity_estimates.get(person_id)
    
    def get_trajectory(self, person_id):
        """Get trajectory history for a person"""
        return list(self.trajectory_history.get(person_id, []))
    
    def remove_person(self, person_id):
        """Remove all motion data for a person"""
        self.kalman_filters.pop(person_id, None)
        self.position_history.pop(person_id, None)
        self.velocity_estimates.pop(person_id, None)
        self.trajectory_history.pop(person_id, None)
    
    def clear(self):
        """Clear all tracking data"""
        self.kalman_filters.clear()
        self.position_history.clear()
        self.velocity_estimates.clear()
        self.trajectory_history.clear()
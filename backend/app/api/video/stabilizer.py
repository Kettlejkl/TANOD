"""
Lessen box passing/jumping and people cross one another 

"""

import numpy as np
from collections import deque
import time


class CrossingAwareStabilizer:
    """
    Enhanced box stabilizer that detects and handles crossing events
    and entry/exit ID reuse scenarios
    
    NOTE: This stabilizer can work independently of ReID crossing detection
    by calculating its own distances from bounding boxes.
    """
    
    def __init__(
        self, 
        ema_alpha=0.35, 
        deadzone=5, 
        max_jump_distance=150,  # Max reasonable movement between frames
        crossing_detection_threshold=180,  # Distance to detect potential crossing
        position_history_length=10,
        track_reuse_window=1.0,  # Time window to detect track ID reuse (seconds)
        entry_exit_jump_threshold=300,  # Larger threshold for entry/exit detection
        use_reid_crossing_info=False  # Whether to trust ReID crossing detection
    ):
        self.base_alpha = ema_alpha
        self.deadzone = deadzone
        self.max_jump_distance = max_jump_distance
        self.crossing_detection_threshold = crossing_detection_threshold
        self.track_reuse_window = track_reuse_window
        self.entry_exit_jump_threshold = entry_exit_jump_threshold
        self.use_reid_crossing_info = use_reid_crossing_info
        
        # Per-track state
        self.smoothed = {}
        self.velocity = {}
        self.position_history = {}  # Track last N positions
        self.last_update_time = {}
        self.crossing_mode = {}  # Track if in crossing mode
        self.position_history_length = position_history_length
        
        # Cross-track awareness
        self.recent_assignments = {}  # track_id -> (persistent_id, timestamp)
        self.assignment_age_threshold = 2.0  # seconds
        
        # Track ID lifecycle tracking
        self.recently_deleted_tracks = {}  # track_id -> (last_position, deletion_time)
        self.track_creation_time = {}  # track_id -> creation_time
        
        print(f"[Stabilizer] Initialized with crossing threshold: {crossing_detection_threshold}px")
        print(f"[Stabilizer] Use ReID crossing info: {use_reid_crossing_info}")
    
    def _get_box_center(self, box):
        """Get center point of bounding box"""
        return np.array([
            box[0] + box[2] / 2,
            box[1] + box[3] / 2
        ])
    
    def _calculate_distance(self, box1, box2):
        """Calculate distance between two boxes"""
        center1 = self._get_box_center(box1)
        center2 = self._get_box_center(box2)
        return np.linalg.norm(center1 - center2)
    
    def _detect_potential_crossing(self, current_boxes):
        """
        Detect if any people are crossing paths
        
        Args:
            current_boxes: dict of track_id -> box
            
        Returns:
            set of track_ids involved in crossing
        """
        crossing_tracks = set()
        
        track_ids = list(current_boxes.keys())
        
        for i, tid1 in enumerate(track_ids):
            for tid2 in track_ids[i+1:]:
                try:
                    distance = self._calculate_distance(
                        current_boxes[tid1],
                        current_boxes[tid2]
                    )
                    
                    if distance < self.crossing_detection_threshold:
                        crossing_tracks.add(tid1)
                        crossing_tracks.add(tid2)
                        print(f"[Stabilizer] 🚶‍♂️🚶‍♀️ Crossing detected: "
                              f"Track {tid1} ↔ {tid2} (dist: {distance:.0f}px)")
                except Exception as e:
                    print(f"[Stabilizer] ⚠️ Error calculating distance: {e}")
                    continue
        
        return crossing_tracks
    
    def _is_position_jump(self, track_id, new_box):
        """
        Detect if new position is a suspicious jump (possible ID swap)
        
        Returns:
            (is_jump, jump_distance, is_likely_track_reuse)
        """
        if track_id not in self.smoothed:
            return False, 0.0, False
        
        prev_box = self.smoothed[track_id]
        distance = self._calculate_distance(prev_box, new_box)
        
        current_time = time.time()
        
        # Check if this track was recently deleted and recreated
        is_likely_reuse = False
        if track_id in self.recently_deleted_tracks:
            old_pos, deletion_time = self.recently_deleted_tracks[track_id]
            time_since_deletion = current_time - deletion_time
            
            if time_since_deletion < self.track_reuse_window:
                # Track ID was just reused - check if position is very different
                distance_from_old = self._calculate_distance(old_pos, new_box)
                
                if distance_from_old > self.entry_exit_jump_threshold:
                    is_likely_reuse = True
                    print(f"[Stabilizer] 🔄 Track {track_id} reused after {time_since_deletion:.2f}s, "
                          f"position change: {distance_from_old:.0f}px")
        
        # Check if jump exceeds reasonable movement
        if distance > self.max_jump_distance:
            return True, distance, is_likely_reuse
        
        return False, distance, is_likely_reuse
    
    def _predict_next_position(self, track_id):
        """Predict next position based on velocity"""
        if track_id not in self.smoothed or track_id not in self.velocity:
            return None
        
        predicted = self.smoothed[track_id].copy()
        predicted[:2] += self.velocity[track_id][:2]
        
        return predicted
    
    def _is_track_new(self, track_id):
        """Check if track was just created (within last few frames)"""
        if track_id not in self.track_creation_time:
            return True
        
        current_time = time.time()
        age = current_time - self.track_creation_time[track_id]
        
        return age < 0.5  # Track is "new" for first 0.5 seconds
    
    def register_assignment(self, track_id, persistent_id):
        """
        Register track_id to persistent_id mapping
        
        This helps detect when ByteTrack reassigns track_ids
        """
        current_time = time.time()
        
        # Check if this track_id was recently used for different person
        if track_id in self.recent_assignments:
            old_pid, old_time = self.recent_assignments[track_id]
            
            if old_pid != persistent_id:
                time_since = current_time - old_time
                
                if time_since < self.assignment_age_threshold:
                    # Track ID reassignment detected!
                    print(f"[Stabilizer] ⚠️ Track {track_id} reassigned: "
                          f"PID {old_pid} -> {persistent_id} "
                          f"(after {time_since:.1f}s)")
                    
                    # Reset smoothing for this track
                    self._reset_track(track_id)
        
        self.recent_assignments[track_id] = (persistent_id, current_time)
    
    def _reset_track(self, track_id):
        """Reset smoothing state for a track"""
        current_time = time.time()
        
        # Store last position before resetting
        if track_id in self.smoothed:
            self.recently_deleted_tracks[track_id] = (
                self.smoothed[track_id].copy(),
                current_time
            )
        
        self.smoothed.pop(track_id, None)
        self.velocity.pop(track_id, None)
        self.position_history.pop(track_id, None)
        self.last_update_time.pop(track_id, None)
        self.crossing_mode.pop(track_id, None)
        self.track_creation_time.pop(track_id, None)
        
        print(f"[Stabilizer] 🔄 Reset track {track_id}")
    
    def filter(self, track_id, box, persistent_id=None, all_current_boxes=None, is_crossing=None):
        """
        Filter bounding box with crossing awareness and entry/exit handling
        
        Args:
            track_id: ByteTrack track ID
            box: [x, y, w, h]
            persistent_id: ReID persistent ID (optional)
            all_current_boxes: dict of all current boxes for crossing detection
                              Format: {track_id: [x, y, w, h], ...}
            is_crossing: Optional boolean to override crossing detection
                        (useful if ReID already detected crossing)
            
        Returns:
            Filtered [x, y, w, h]
        """
        box = np.array(box, dtype=float)
        current_time = time.time()
        
        # Register assignment if provided
        if persistent_id is not None:
            self.register_assignment(track_id, persistent_id)
        
        # Detect crossing situation
        crossing_tracks = set()
        if all_current_boxes is not None and len(all_current_boxes) > 1:
            try:
                crossing_tracks = self._detect_potential_crossing(all_current_boxes)
            except Exception as e:
                print(f"[Stabilizer] ⚠️ Crossing detection failed: {e}")
        
        # Use provided crossing info if available
        if is_crossing is not None:
            if is_crossing:
                crossing_tracks.add(track_id)
        
        is_crossing_active = track_id in crossing_tracks
        
        if is_crossing_active and not self.crossing_mode.get(track_id, False):
            print(f"[Stabilizer] 🚶‍♂️🚶‍♀️ Track {track_id} entering crossing mode")
        
        # First detection of this track
        if track_id not in self.smoothed:
            self.smoothed[track_id] = box
            self.velocity[track_id] = np.zeros(4)
            self.position_history[track_id] = deque([box.copy()], maxlen=self.position_history_length)
            self.last_update_time[track_id] = current_time
            self.crossing_mode[track_id] = is_crossing_active
            self.track_creation_time[track_id] = current_time
            
            # Clean up old deletion record if exists
            self.recently_deleted_tracks.pop(track_id, None)
            
            print(f"[Stabilizer] 🆕 New track {track_id} at position {box[:2].astype(int)}")
            
            return box.astype(int)
        
        # Check for suspicious position jump
        is_jump, jump_distance, is_likely_reuse = self._is_position_jump(track_id, box)
        
        # Handle track ID reuse (entry/exit scenario)
        if is_likely_reuse:
            print(f"[Stabilizer] 🚪 Entry/exit track reuse detected for {track_id}")
            self._reset_track(track_id)
            self.smoothed[track_id] = box
            self.velocity[track_id] = np.zeros(4)
            self.position_history[track_id] = deque([box.copy()], maxlen=self.position_history_length)
            self.last_update_time[track_id] = current_time
            self.crossing_mode[track_id] = is_crossing_active
            self.track_creation_time[track_id] = current_time
            return box.astype(int)
        
        # Handle other position jumps
        if is_jump:
            print(f"[Stabilizer] ⚠️ Position jump detected for track {track_id}: "
                  f"{jump_distance:.0f}px")
            
            # Large jump during crossing = likely ID swap
            if is_crossing_active:
                print(f"[Stabilizer] 🔄 Resetting track {track_id} due to crossing jump")
                self._reset_track(track_id)
                self.smoothed[track_id] = box
                self.velocity[track_id] = np.zeros(4)
                self.position_history[track_id] = deque([box.copy()], maxlen=self.position_history_length)
                self.last_update_time[track_id] = current_time
                self.crossing_mode[track_id] = True
                self.track_creation_time[track_id] = current_time
                return box.astype(int)
            
            # For new tracks, accept the jump (might be legitimate detection improvement)
            if self._is_track_new(track_id):
                print(f"[Stabilizer] 📍 Accepting jump for new track {track_id}")
                self.smoothed[track_id] = box
                self.velocity[track_id] = np.zeros(4)
                return box.astype(int)
        
        # Get previous state
        prev_box = self.smoothed[track_id]
        time_delta = current_time - self.last_update_time[track_id]
        
        # Calculate raw difference
        raw_diff = box - prev_box
        
        # Apply deadzone to ignore minor jitter
        raw_diff = np.where(np.abs(raw_diff) < self.deadzone, 0, raw_diff)
        
        # Update velocity with smoothing
        new_velocity = raw_diff * 0.6 + self.velocity[track_id] * 0.4
        self.velocity[track_id] = new_velocity
        
        # Adjust smoothing based on crossing state and track age
        is_new_track = self._is_track_new(track_id)
        
        if is_new_track:
            # New tracks: trust detection more, smooth less
            alpha = 0.7
        elif is_crossing_active:
            # During crossing: significantly reduce smoothing to track raw detections
            # This prevents boxes from "sticking" to wrong person during crossings
            alpha = 0.75  # High alpha = trust current detection more
            
            if not self.crossing_mode.get(track_id, False):
                print(f"[Stabilizer] 🚶 Track {track_id} entering crossing mode (alpha={alpha:.2f})")
            
            self.crossing_mode[track_id] = True
        else:
            # Normal mode: more smoothing
            alpha = self.base_alpha
            
            if self.crossing_mode.get(track_id, False):
                print(f"[Stabilizer] ✅ Track {track_id} exiting crossing mode")
            
            self.crossing_mode[track_id] = False
        
        # Check for erratic movement (possible tracking error)
        velocity_magnitude = np.linalg.norm(new_velocity[:2])
        
        if velocity_magnitude > 100:  # Very fast movement
            # Possible tracking error - trust raw detection
            print(f"[Stabilizer] ⚡ High velocity ({velocity_magnitude:.0f}px) for track {track_id}")
            smoothed_box = box
        else:
            # Normal smoothing with velocity prediction
            prediction = prev_box + new_velocity * 0.8
            smoothed_box = alpha * box + (1 - alpha) * prediction
        
        # Update state
        self.smoothed[track_id] = smoothed_box
        self.position_history[track_id].append(smoothed_box.copy())
        self.last_update_time[track_id] = current_time
        
        return smoothed_box.astype(int)
    
    def cleanup(self, active_ids):
        """Remove tracks that are no longer active"""
        current_time = time.time()
        
        # Store positions of tracks being deleted
        for track_id in list(self.smoothed.keys()):
            if track_id not in active_ids:
                if track_id in self.smoothed:
                    self.recently_deleted_tracks[track_id] = (
                        self.smoothed[track_id].copy(),
                        current_time
                    )
                    print(f"[Stabilizer] 🗑️ Track {track_id} removed, storing position for reuse detection")
        
        # Clean smoothed tracks
        self.smoothed = {k: v for k, v in self.smoothed.items() if k in active_ids}
        self.velocity = {k: v for k, v in self.velocity.items() if k in active_ids}
        self.position_history = {k: v for k, v in self.position_history.items() if k in active_ids}
        self.last_update_time = {k: v for k, v in self.last_update_time.items() if k in active_ids}
        self.crossing_mode = {k: v for k, v in self.crossing_mode.items() if k in active_ids}
        self.track_creation_time = {k: v for k, v in self.track_creation_time.items() if k in active_ids}
        
        # Clean old assignments
        expired_assignments = [
            tid for tid, (pid, timestamp) in self.recent_assignments.items()
            if current_time - timestamp > self.assignment_age_threshold * 2
        ]
        
        for tid in expired_assignments:
            del self.recent_assignments[tid]
        
        # Clean old deleted track records
        expired_deletions = [
            tid for tid, (pos, timestamp) in self.recently_deleted_tracks.items()
            if current_time - timestamp > self.track_reuse_window * 2
        ]
        
        for tid in expired_deletions:
            del self.recently_deleted_tracks[tid]
    
    def get_status(self):
        """Get stabilizer status for debugging"""
        crossing_count = sum(1 for is_crossing in self.crossing_mode.values() if is_crossing)
        
        return {
            'active_tracks': len(self.smoothed),
            'crossing_tracks': crossing_count,
            'recent_assignments': len(self.recent_assignments),
            'recently_deleted': len(self.recently_deleted_tracks),
            'max_jump_distance': self.max_jump_distance,
            'crossing_threshold': self.crossing_detection_threshold,
            'entry_exit_threshold': self.entry_exit_jump_threshold
        }


# Backward compatible alias
ResponsiveBoxFilter = CrossingAwareStabilizer
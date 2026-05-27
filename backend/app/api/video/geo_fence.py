# app/api/video/geo_fence.py
import cv2
import numpy as np
import uuid

class GeoFence:
    """Single 2D geo-fence polygon"""
    def __init__(self, fence_id=None, name="Zone", points=None, enabled=True):
        self.id = fence_id or str(uuid.uuid4())
        self.name = name
        self.points = np.array(points, dtype=np.int32) if points else np.array([])
        self.enabled = enabled
    
    def set_points(self, points):
        """Set the corner points of the geo-fence (minimum 3 points for polygon)"""
        if len(points) < 3:
            print(f"[GeoFence] Error: Expected at least 3 points, got {len(points)}")
            return False
            
        formatted_points = []
        for p in points:
            try:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    # Handle both [x, y] and [x, y, confidence] formats
                    formatted_points.append([int(float(p[0])), int(float(p[1]))])
                elif isinstance(p, dict) and 'x' in p and 'y' in p:
                    # Handle {x: ..., y: ...} format
                    formatted_points.append([int(float(p['x'])), int(float(p['y']))])
                else:
                    print(f"[GeoFence] Error: Invalid point format: {p}")
                    return False
            except (ValueError, TypeError) as e:
                print(f"[GeoFence] Error converting point {p}: {e}")
                return False
        
        if len(formatted_points) < 3:
            print(f"[GeoFence] Error: After formatting, got {len(formatted_points)} valid points (need 3+)")
            return False
        
        self.points = np.array(formatted_points, dtype=np.int32)
        print(f"[GeoFence] Set fence '{self.name}' with {len(formatted_points)} points: {self.points.tolist()}")
        return True
    
    def get_points(self):
        """Get points as list of lists for JSON serialization"""
        if len(self.points) > 0:
            return self.points.tolist()
        return []
    
    def get_config(self):
        """Get full configuration"""
        return {
            'id': self.id,
            'name': self.name,
            'points': self.get_points(),
            'enabled': self.enabled
        }
    
    def is_inside(self, x, y):
        """Check if a point (x, y) is inside the geo-fence"""
        if not self.enabled or len(self.points) < 3:
            return False
        
        point = (float(x), float(y))
        result = cv2.pointPolygonTest(self.points, point, False)
        return result >= 0
    
    def is_person_inside(self, box):
        """Check if person's center point is inside the geo-fence"""
        l, t, r, b = box
        center_x = (l + r) // 2
        center_y = (t + b) // 2
        
        return self.is_inside(center_x, center_y)


class MultiGeoFenceManager:
    """Manages multiple geo-fences for a single camera"""
    def __init__(self):
        self.fences = []  # List of GeoFence objects
    
    def add_fence(self, name, points):
        """Add a new geo-fence"""
        if not name or not name.strip():
            name = f"Zone {len(self.fences) + 1}"
        
        fence = GeoFence(name=name.strip(), points=[], enabled=True)
        if fence.set_points(points):
            self.fences.append(fence)
            print(f"[MultiGeoFenceManager] Added fence '{fence.name}' with ID {fence.id}")
            return fence.id
        else:
            print(f"[MultiGeoFenceManager] Failed to add fence '{name}' - invalid points")
            return None
    
    def remove_fence(self, fence_id):
        """Remove a geo-fence by ID"""
        initial_count = len(self.fences)
        self.fences = [f for f in self.fences if f.id != fence_id]
        removed = len(self.fences) < initial_count
        
        if removed:
            print(f"[MultiGeoFenceManager] Removed fence with ID {fence_id}")
        else:
            print(f"[MultiGeoFenceManager] Fence with ID {fence_id} not found")
        
        return removed
    
    def update_fence(self, fence_id, points=None, name=None, enabled=None):
        """Update an existing geo-fence"""
        fence = self.get_fence(fence_id)
        if not fence:
            print(f"[MultiGeoFenceManager] Fence {fence_id} not found for update")
            return False
        
        try:
            if points is not None:
                if not fence.set_points(points):
                    print(f"[MultiGeoFenceManager] Failed to update points for fence {fence_id}")
                    return False
            
            if name is not None:
                if name.strip():
                    fence.name = name.strip()
                    print(f"[MultiGeoFenceManager] Updated fence name to '{fence.name}'")
                else:
                    print(f"[MultiGeoFenceManager] Invalid name (empty), keeping '{fence.name}'")
            
            if enabled is not None:
                fence.enabled = bool(enabled)
                status = "enabled" if fence.enabled else "disabled"
                print(f"[MultiGeoFenceManager] Fence '{fence.name}' {status}")
            
            print(f"[MultiGeoFenceManager] Successfully updated fence '{fence.name}' (ID: {fence_id})")
            return True
            
        except Exception as e:
            print(f"[MultiGeoFenceManager] Error updating fence {fence_id}: {e}")
            return False
    
    def toggle_fence(self, fence_id):
        """Toggle fence enabled/disabled"""
        fence = self.get_fence(fence_id)
        if not fence:
            print(f"[MultiGeoFenceManager] Fence {fence_id} not found for toggle")
            return None
        
        fence.enabled = not fence.enabled
        status = "enabled" if fence.enabled else "disabled"
        print(f"[MultiGeoFenceManager] Fence '{fence.name}' {status}")
        return fence.enabled
    
    def get_fence(self, fence_id):
        """Get a specific fence by ID"""
        for fence in self.fences:
            if fence.id == fence_id:
                return fence
        return None
    
    def get_all_fences(self):
        """Get all fences configuration"""
        configs = [fence.get_config() for fence in self.fences]
        print(f"[MultiGeoFenceManager] Returning {len(configs)} fence configs")
        return configs
    
    def is_person_inside_any(self, box):
        """Check if person is inside any enabled fence"""
        for fence in self.fences:
            if fence.enabled and fence.is_person_inside(box):
                return True, fence.id, fence.name
        return False, None, None
    
    def count_inside_fences(self, box):
        """Count how many enabled fences the person is inside"""
        count = 0
        fence_ids = []
        for fence in self.fences:
            if fence.enabled and fence.is_person_inside(box):
                count += 1
                fence_ids.append(fence.id)
        return count, fence_ids
    
    def clear_all(self):
        """Remove all fences"""
        count = len(self.fences)
        self.fences = []
        print(f"[MultiGeoFenceManager] Cleared all {count} fences")
        return count
    
    def load_from_config(self, fences_config):
        """Load fences from configuration list"""
        self.fences = []
        loaded_count = 0
        
        if not isinstance(fences_config, list):
            print(f"[MultiGeoFenceManager] Error: Expected list, got {type(fences_config)}")
            return False
        
        for config in fences_config:
            try:
                fence_id = config.get('id')
                name = config.get('name', 'Zone')
                points = config.get('points', [])
                enabled = config.get('enabled', True)
                
                if len(points) < 3:
                    print(f"[MultiGeoFenceManager] Skipping fence '{name}' - insufficient points")
                    continue
                
                fence = GeoFence(
                    fence_id=fence_id,
                    name=name,
                    points=[],
                    enabled=enabled
                )
                
                if fence.set_points(points):
                    self.fences.append(fence)
                    loaded_count += 1
                else:
                    print(f"[MultiGeoFenceManager] Failed to load fence '{name}'")
                    
            except Exception as e:
                print(f"[MultiGeoFenceManager] Error loading fence config: {e}")
                continue
        
        print(f"[MultiGeoFenceManager] Loaded {loaded_count}/{len(fences_config)} fences from config")
        return loaded_count > 0
import cv2
import time
import threading
from datetime import datetime


class SynchronizedVideoManager:
    def __init__(self):
        self.cameras = {}
        self.sync_lock = threading.Lock()
        self.master_clock = 0.0
        self.playback_start_time = None
        self.is_paused = False
        self.playback_speed = 1.0
        
    def add_camera(self, camera_id, cap, start_offset_sec=0.0):
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        self.cameras[camera_id] = {
            'cap': cap,
            'fps': fps,
            'total_frames': total_frames,
            'duration_sec': duration_sec,
            'start_offset_sec': start_offset_sec,
        }
        
        print(f"✅ Sync enabled for {camera_id}: {duration_sec/60:.1f}min, offset={start_offset_sec}s")
        
    def initialize_master_clock(self):
        self.playback_start_time = time.time()
        self.master_clock = 0.0
        print(f"🕐 Master clock initialized at {datetime.now().strftime('%H:%M:%S')}")
        
    def get_master_time(self):
        if self.is_paused or self.playback_start_time is None:
            return self.master_clock
        elapsed = (time.time() - self.playback_start_time) * self.playback_speed
        return self.master_clock + elapsed
    
    def get_target_frame(self, camera_id):
        if camera_id not in self.cameras:
            return None, 0.0
            
        cam = self.cameras[camera_id]
        master_time = self.get_master_time()
        
        camera_time = master_time - cam['start_offset_sec']
        
        if camera_time < 0:
            return 0, master_time
        if camera_time > cam['duration_sec']:
            camera_time = camera_time % cam['duration_sec']
        
        target_frame = int(camera_time * cam['fps'])
        target_frame = min(target_frame, cam['total_frames'] - 1)
        
        return target_frame, master_time
    
    def reset(self):
        self.master_clock = 0.0
        self.playback_start_time = time.time()
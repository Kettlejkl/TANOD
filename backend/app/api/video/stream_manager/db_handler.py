class AnalyticsDBHandler:
    def __init__(self, analytics_db_instance):
        self.db = analytics_db_instance
    
    def save_person_track(self, camera_id, persistent_id, track_id, bbox, metadata=None):
        if self.db is None:
            return
        try:
            detection_id = self.db.save_detection(
                persistent_id=persistent_id,
                camera_id=camera_id,
                track_id=track_id,
                bbox=bbox,
                confidence=metadata.get('confidence', 0.8) if metadata else 0.8,
                in_geo_fence=metadata.get('in_geo_fence', False) if metadata else False,
                fence_id=metadata.get('fence_id') if metadata else None,
                fence_name=metadata.get('fence_name') if metadata else None,
                frame_id=metadata.get('frame_id', 0) if metadata else 0
            )
            return detection_id
        except Exception as e:
            print(f"[AnalyticsDBHandler] Error saving detection: {e}")
            return None
    
    def save_behavior_event(self, camera_id, persistent_id, track_id, behavior_type, 
                           severity, confidence, description, metadata=None, position=None):
        if self.db is None:
            return
        try:
            detections = self.db.get_detections(
                camera_id=camera_id,
                persistent_id=persistent_id,
                limit=1
            )
            if not detections:
                print(f"[AnalyticsDBHandler] No detection found for UID {persistent_id}")
                return None
            detection_id = detections[0]['id']
            behavior_id = self.db.save_behavior(
                detection_id=detection_id,
                behavior_type=behavior_type,
                severity=severity,
                confidence=confidence,
                description=description,
                metadata=metadata,
                position=position
            )
            self.db.update_person_journey(persistent_id)
            return behavior_id
        except Exception as e:
            print(f"[AnalyticsDBHandler] Error saving behavior: {e}")
            return None
    
    def deactivate_tracks(self, camera_id, active_ids):
        if self.db is None:
            return
        try:
            detections = self.db.get_detections(camera_id=camera_id, limit=1000)
            all_pids = set(d['persistent_id'] for d in detections)
            inactive_pids = all_pids - set(active_ids)
            for pid in inactive_pids:
                self.db.close_person_journey(pid)
        except Exception as e:
            print(f"[AnalyticsDBHandler] Error deactivating tracks: {e}")
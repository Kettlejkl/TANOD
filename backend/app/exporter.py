# app/api/analytics/exporter.py
"""
Export analytics data to CSV files using pandas
Aggregates data by persistent_id instead of per-frame
"""

import pandas as pd
from datetime import datetime, timedelta
import os

from .database import analytics_db


class AnalyticsExporter:
    """
    Handles exporting analytics data to CSV files
    """
    
    def __init__(self, export_dir='exports'):
        """
        Initialize exporter
        
        Args:
            export_dir: Directory to save CSV files
        """
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
        print(f"[AnalyticsExporter] ✅ Export directory: {export_dir}")
    
    # ============================================================================
    # PERSON-LEVEL EXPORTS (ONE ROW PER PERSISTENT_ID)
    # ============================================================================
    
    def export_persons_summary(self, camera_id=None, start_time=None, end_time=None,
                               filename=None):
        """
        Export person summary (one row per persistent_id)
        
        Args:
            camera_id: Filter by camera ID (None = all cameras)
            start_time: Start timestamp
            end_time: End timestamp
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to exported CSV file
        """
        # Get all detections
        detections = analytics_db.get_detections(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=1000000  # Get all
        )
        
        if not detections:
            print("[AnalyticsExporter] ⚠️ No detections to export")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(detections)
        
        # Aggregate by persistent_id
        summary = df.groupby('persistent_id').agg({
            'camera_id': lambda x: ','.join(sorted(set(x))),  # Cameras visited
            'track_id': 'first',
            'confidence': 'mean',
            'in_geo_fence': lambda x: (x.sum() / len(x) * 100),  # % time in fence
            'fence_id': lambda x: ','.join(sorted(set(str(v) for v in x if v))),
            'fence_name': lambda x: ','.join(sorted(set(str(v) for v in x if v))),
            'frame_id': 'count',  # Total frames detected
            'timestamp': ['min', 'max']  # First and last seen
        }).reset_index()
        
        # Flatten column names
        summary.columns = [
            'persistent_id',
            'cameras_visited',
            'track_id',
            'avg_confidence',
            'percent_time_in_fence',
            'fence_ids_visited',
            'fence_names_visited',
            'total_detections',
            'first_seen',
            'last_seen'
        ]
        
        # Calculate duration
        summary['duration_seconds'] = (
            pd.to_datetime(summary['last_seen']) - 
            pd.to_datetime(summary['first_seen'])
        ).dt.total_seconds()
        
        # Round percentages
        summary['percent_time_in_fence'] = summary['percent_time_in_fence'].round(2)
        summary['avg_confidence'] = summary['avg_confidence'].round(3)
        
        # Sort by first_seen
        summary = summary.sort_values('first_seen')
        
        # Generate filename if not provided
        if not filename:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            cam_str = camera_id if camera_id else 'all_cameras'
            filename = f'persons_summary_{cam_str}_{timestamp_str}.csv'
        
        # Export to CSV
        filepath = os.path.join(self.export_dir, filename)
        summary.to_csv(filepath, index=False)
        
        print(f"[AnalyticsExporter] ✅ Exported {len(summary)} persons to {filepath}")
        return filepath
    
    def export_behaviors_by_person(self, camera_id=None, start_time=None, end_time=None,
                                   filename=None):
        """
        Export behaviors aggregated by person (one row per persistent_id)
        
        Args:
            camera_id: Filter by camera ID
            start_time: Start timestamp
            end_time: End timestamp
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to exported CSV file
        """
        # Get all behaviors with detection info
        behaviors = analytics_db.get_behaviors(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=1000000
        )
        
        if not behaviors:
            print("[AnalyticsExporter] ⚠️ No behaviors to export")
            return None
        
        # Extract data
        data = []
        for b in behaviors:
            data.append({
                'persistent_id': b['detection']['persistent_id'],
                'camera_id': b['detection']['camera_id'],
                'behavior_type': b['behavior_type'],
                'severity': b['severity'],
                'confidence': b['confidence'],
                'timestamp': b['timestamp']
            })
        
        df = pd.DataFrame(data)
        
        # Aggregate by persistent_id
        summary = df.groupby('persistent_id').agg({
            'camera_id': lambda x: ','.join(sorted(set(x))),
            'behavior_type': lambda x: ','.join(sorted(set(x))),
            'severity': lambda x: max(x, key=lambda v: ['low', 'medium', 'high'].index(v)),
            'confidence': 'mean',
            'timestamp': ['min', 'max', 'count']
        }).reset_index()
        
        # Flatten columns
        summary.columns = [
            'persistent_id',
            'cameras',
            'behavior_types',
            'max_severity',
            'avg_confidence',
            'first_behavior',
            'last_behavior',
            'total_behaviors'
        ]
        
        # Count each behavior type
        behavior_counts = df.groupby(['persistent_id', 'behavior_type']).size().unstack(fill_value=0)
        behavior_counts.columns = [f'{col}_count' for col in behavior_counts.columns]
        
        # Merge counts
        summary = summary.merge(behavior_counts, on='persistent_id', how='left')
        
        # Round confidence
        summary['avg_confidence'] = summary['avg_confidence'].round(3)
        
        # Sort by total behaviors
        summary = summary.sort_values('total_behaviors', ascending=False)
        
        # Generate filename
        if not filename:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            cam_str = camera_id if camera_id else 'all_cameras'
            filename = f'behaviors_by_person_{cam_str}_{timestamp_str}.csv'
        
        # Export
        filepath = os.path.join(self.export_dir, filename)
        summary.to_csv(filepath, index=False)
        
        print(f"[AnalyticsExporter] ✅ Exported behaviors for {len(summary)} persons to {filepath}")
        return filepath
    
    # ============================================================================
    # KEEP EXISTING METHODS FOR COMPATIBILITY
    # ============================================================================
    
    def export_detections(self, camera_id=None, start_time=None, end_time=None,
                         filename=None):
        """
        Export raw detections (per frame) - kept for compatibility
        """
        detections = analytics_db.get_detections(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=100000
        )
        
        if not detections:
            print("[AnalyticsExporter] ⚠️ No detections to export")
            return None
        
        df = pd.DataFrame(detections)
        
        # Flatten bbox
        df['bbox_x'] = df['bbox'].apply(lambda x: x['x'])
        df['bbox_y'] = df['bbox'].apply(lambda x: x['y'])
        df['bbox_width'] = df['bbox'].apply(lambda x: x['width'])
        df['bbox_height'] = df['bbox'].apply(lambda x: x['height'])
        df = df.drop('bbox', axis=1)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if not filename:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            cam_str = camera_id if camera_id else 'all_cameras'
            filename = f'detections_{cam_str}_{timestamp_str}.csv'
        
        filepath = os.path.join(self.export_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"[AnalyticsExporter] ✅ Exported {len(df)} detections to {filepath}")
        return filepath
    
    def export_behaviors(self, camera_id=None, behavior_type=None, severity=None,
                        start_time=None, end_time=None, filename=None):
        """
        Export raw behaviors (per event) - kept for compatibility
        """
        behaviors = analytics_db.get_behaviors(
            camera_id=camera_id,
            behavior_type=behavior_type,
            severity=severity,
            start_time=start_time,
            end_time=end_time,
            limit=100000
        )
        
        if not behaviors:
            print("[AnalyticsExporter] ⚠️ No behaviors to export")
            return None
        
        flattened = []
        for b in behaviors:
            row = {
                'behavior_id': b['id'],
                'persistent_id': b['detection']['persistent_id'],
                'camera_id': b['detection']['camera_id'],
                'behavior_type': b['behavior_type'],
                'severity': b['severity'],
                'confidence': b['confidence'],
                'description': b['description'],
                'timestamp': b['timestamp']
            }
            flattened.append(row)
        
        df = pd.DataFrame(flattened)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if not filename:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            type_str = behavior_type if behavior_type else 'all_behaviors'
            filename = f'behaviors_{type_str}_{timestamp_str}.csv'
        
        filepath = os.path.join(self.export_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"[AnalyticsExporter] ✅ Exported {len(df)} behaviors to {filepath}")
        return filepath
    
    def export_hourly_stats(self, camera_id, start_time=None, end_time=None,
                           filename=None):
        """Export hourly statistics"""
        stats = analytics_db.get_hourly_stats(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time
        )
        
        if not stats:
            print("[AnalyticsExporter] ⚠️ No hourly stats to export")
            return None
        
        df = pd.DataFrame(stats)
        df['hour'] = pd.to_datetime(df['hour'])
        
        if not filename:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'hourly_stats_{camera_id}_{timestamp_str}.csv'
        
        filepath = os.path.join(self.export_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"[AnalyticsExporter] ✅ Exported {len(df)} hourly stats to {filepath}")
        return filepath
    
    def export_daily_reports(self, camera_id, start_date=None, end_date=None,
                            filename=None):
        """Export daily reports"""
        reports = analytics_db.get_daily_reports(
            camera_id=camera_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not reports:
            print("[AnalyticsExporter] ⚠️ No daily reports to export")
            return None
        
        df = pd.DataFrame(reports)
        df['date'] = pd.to_datetime(df['date'])
        
        if not filename:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'daily_reports_{camera_id}_{timestamp_str}.csv'
        
        filepath = os.path.join(self.export_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"[AnalyticsExporter] ✅ Exported {len(df)} daily reports to {filepath}")
        return filepath
    
    def export_person_journeys(self, start_time=None, end_time=None, 
                              is_active=None, filename=None):
        """Export person journeys"""
        with analytics_db.session_scope() as session:
            from .models import PersonJourney
            
            query = session.query(PersonJourney)
            
            if start_time:
                query = query.filter(PersonJourney.first_seen >= start_time)
            if end_time:
                query = query.filter(PersonJourney.first_seen <= end_time)
            if is_active is not None:
                query = query.filter(PersonJourney.is_active == is_active)
            
            journeys = [j.to_dict() for j in query.all()]
        
        if not journeys:
            print("[AnalyticsExporter] ⚠️ No journeys to export")
            return None
        
        flattened = []
        for j in journeys:
            row = {
                'journey_id': j['id'],
                'persistent_id': j['persistent_id'],
                'first_seen': j['first_seen'],
                'last_seen': j['last_seen'],
                'total_duration_seconds': j['total_duration_seconds'],
                'total_behaviors': j['total_behaviors'],
                'time_in_fence_seconds': j['time_in_fence_seconds'],
                'time_outside_fence_seconds': j['time_outside_fence_seconds'],
                'is_active': j['is_active'],
                'cameras_visited': ','.join(j['cameras_visited']) if j['cameras_visited'] else '',
                'behavior_types': ','.join(j['behavior_types']) if j['behavior_types'] else ''
            }
            flattened.append(row)
        
        df = pd.DataFrame(flattened)
        df['first_seen'] = pd.to_datetime(df['first_seen'])
        df['last_seen'] = pd.to_datetime(df['last_seen'])
        
        if not filename:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'person_journeys_{timestamp_str}.csv'
        
        filepath = os.path.join(self.export_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"[AnalyticsExporter] ✅ Exported {len(df)} journeys to {filepath}")
        return filepath
    
    # ============================================================================
    # NEW: AGGREGATED BATCH EXPORT
    # ============================================================================
    
    def export_all_data(self, camera_id, start_time, end_time, prefix=''):
        """
        Export all analytics data (person-level aggregation)
        
        Args:
            camera_id: Camera identifier
            start_time: Start timestamp
            end_time: End timestamp
            prefix: Optional prefix for all filenames
        
        Returns:
            Dict with paths to all exported files
        """
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f"{prefix}_" if prefix else ""
        
        exports = {}
        
        # Export person summary (aggregated by persistent_id)
        exports['persons_summary'] = self.export_persons_summary(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            filename=f'{prefix}persons_summary_{camera_id}_{timestamp_str}.csv'
        )
        
        # Export behaviors by person
        exports['behaviors_by_person'] = self.export_behaviors_by_person(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            filename=f'{prefix}behaviors_by_person_{camera_id}_{timestamp_str}.csv'
        )
        
        # Export hourly stats
        exports['hourly_stats'] = self.export_hourly_stats(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            filename=f'{prefix}hourly_stats_{camera_id}_{timestamp_str}.csv'
        )
        
        # Export person journeys (cross-camera)
        exports['journeys'] = self.export_person_journeys(
            start_time=start_time,
            end_time=end_time,
            filename=f'{prefix}journeys_{timestamp_str}.csv'
        )
        
        print(f"[AnalyticsExporter] ✅ Batch export complete for {camera_id}!")
        return exports


# Global instance
exporter = AnalyticsExporter()
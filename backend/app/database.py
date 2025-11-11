"""
Database handler for analytics using SQLAlchemy
"""

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from datetime import datetime, timedelta
import os

from .models import Base, PersonDetection, BehaviorEvent, CameraStatistics, PersonJourney, DailyReport


class AnalyticsDatabase:
    """
    Handles all database operations for analytics
    """
    
    def __init__(self, db_path='analytics.db'):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Session factory
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)
        
        print(f"[AnalyticsDB] ✅ Database initialized at {db_path}")
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for database operations"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"[AnalyticsDB] ❌ Error: {e}")
            raise
        finally:
            session.close()
    
    # ============================================================================
    # PERSON DETECTION OPERATIONS
    # ============================================================================
    
    def save_detection(self, persistent_id, camera_id, track_id, bbox, confidence,
                      in_geo_fence=False, fence_id=None, fence_name=None,
                      frame_id=0, timestamp=None):
        """
        Save a person detection to database
        
        Args:
            persistent_id: Unique person ID (UID)
            camera_id: Camera identifier
            track_id: Tracker ID
            bbox: Bounding box [x, y, width, height]
            confidence: Detection confidence (0-1)
            in_geo_fence: Whether person is in geo-fence
            fence_id: Geo-fence ID (if in fence)
            fence_name: Geo-fence name (if in fence)
            frame_id: Frame number
            timestamp: Detection timestamp (defaults to now)
        
        Returns:
            PersonDetection object
        """
        with self.session_scope() as session:
            detection = PersonDetection(
                persistent_id=persistent_id,
                camera_id=camera_id,
                track_id=track_id,
                bbox_x=bbox[0],
                bbox_y=bbox[1],
                bbox_width=bbox[2],
                bbox_height=bbox[3],
                confidence=confidence,
                in_geo_fence=in_geo_fence,
                fence_id=fence_id,
                fence_name=fence_name,
                frame_id=frame_id,
                timestamp=timestamp or datetime.utcnow()
            )
            session.add(detection)
            session.flush()  # Get ID before commit
            return detection.id
    
    def get_detections(self, camera_id=None, start_time=None, end_time=None, 
                      persistent_id=None, limit=1000):
        """
        Query person detections with filters
        
        Args:
            camera_id: Filter by camera ID
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            persistent_id: Filter by person UID
            limit: Maximum results to return
        
        Returns:
            List of PersonDetection objects
        """
        with self.session_scope() as session:
            query = session.query(PersonDetection)
            
            if camera_id:
                query = query.filter(PersonDetection.camera_id == camera_id)
            if start_time:
                query = query.filter(PersonDetection.timestamp >= start_time)
            if end_time:
                query = query.filter(PersonDetection.timestamp <= end_time)
            if persistent_id is not None:
                query = query.filter(PersonDetection.persistent_id == persistent_id)
            
            query = query.order_by(PersonDetection.timestamp.desc())
            query = query.limit(limit)
            
            return [d.to_dict() for d in query.all()]
    
    # ============================================================================
    # BEHAVIOR EVENT OPERATIONS
    # ============================================================================
    
    def save_behavior(self, detection_id, behavior_type, severity, confidence,
                     description=None, metadata=None, position=None, timestamp=None):
        """
        Save a behavior event to database
        
        Args:
            detection_id: Associated detection ID
            behavior_type: Type of behavior (loitering, running, suspicious, crowd, violence, fallen, fire, smoke)
            severity: Severity level (low, medium, high)
            confidence: Detection confidence (0-1)
            description: Human-readable description
            metadata: Additional behavior-specific data (dict)
            position: Position tuple (x, y) or None
            timestamp: Event timestamp (defaults to now)
        
        Returns:
            BehaviorEvent ID
        """
        with self.session_scope() as session:
            behavior = BehaviorEvent(
                detection_id=detection_id,
                behavior_type=behavior_type,
                severity=severity,
                confidence=confidence,
                description=description,
                metadata=metadata,
                position_x=position[0] if position else None,
                position_y=position[1] if position else None,
                timestamp=timestamp or datetime.utcnow()
            )
            session.add(behavior)
            session.flush()
            return behavior.id
    
    def get_behaviors(self, camera_id=None, behavior_type=None, severity=None,
                     start_time=None, end_time=None, limit=1000):
        """
        Query behavior events with filters
        
        Args:
            camera_id: Filter by camera ID
            behavior_type: Filter by behavior type
            severity: Filter by severity
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum results to return
        
        Returns:
            List of BehaviorEvent dicts with detection info
        """
        with self.session_scope() as session:
            query = session.query(BehaviorEvent).join(PersonDetection)
            
            if camera_id:
                query = query.filter(PersonDetection.camera_id == camera_id)
            if behavior_type:
                query = query.filter(BehaviorEvent.behavior_type == behavior_type)
            if severity:
                query = query.filter(BehaviorEvent.severity == severity)
            if start_time:
                query = query.filter(BehaviorEvent.timestamp >= start_time)
            if end_time:
                query = query.filter(BehaviorEvent.timestamp <= end_time)
            
            query = query.order_by(BehaviorEvent.timestamp.desc())
            query = query.limit(limit)
            
            results = []
            for behavior in query.all():
                result = behavior.to_dict()
                result['detection'] = behavior.detection.to_dict()
                results.append(result)
            
            return results
    
    # ============================================================================
    # STATISTICS OPERATIONS
    # ============================================================================

    def update_hourly_stats(self, camera_id, hour_timestamp):
        """
        Calculate and update hourly statistics for a camera
        Supports new behavior types (violence, fallen, fire, smoke)
        
        Args:
            camera_id: Camera identifier
            hour_timestamp: Hour timestamp (rounded to hour)
        """
        with self.session_scope() as session:
            # Round to hour
            hour_start = hour_timestamp.replace(minute=0, second=0, microsecond=0)
            hour_end = hour_start + timedelta(hours=1)
            
            # Get or create stats record
            stats = session.query(CameraStatistics).filter(
                CameraStatistics.camera_id == camera_id,
                CameraStatistics.hour == hour_start
            ).first()
            
            if not stats:
                stats = CameraStatistics(camera_id=camera_id, hour=hour_start)
                session.add(stats)
            
            # Count detections
            detections = session.query(PersonDetection).filter(
                PersonDetection.camera_id == camera_id,
                PersonDetection.timestamp >= hour_start,
                PersonDetection.timestamp < hour_end
            ).all()
            
            stats.total_detections = len(detections)
            stats.unique_persons = len(set(d.persistent_id for d in detections))
            
            # Calculate average persons per frame
            if detections:
                frames = set(d.frame_id for d in detections)
                stats.avg_persons_per_frame = len(detections) / len(frames) if frames else 0
            
            # Calculate peak persons (max in single frame)
            if detections:
                from collections import Counter
                frame_counts = Counter(d.frame_id for d in detections)
                stats.peak_persons = max(frame_counts.values())
            
            # Count behaviors
            behaviors = session.query(BehaviorEvent).join(PersonDetection).filter(
                PersonDetection.camera_id == camera_id,
                BehaviorEvent.timestamp >= hour_start,
                BehaviorEvent.timestamp < hour_end
            ).all()
            
            # Reset counts
            stats.loitering_events = 0
            stats.running_events = 0
            stats.violence_events = 0
            stats.fallen_events = 0
            stats.crowd_events = 0
            stats.fire_events = 0
            stats.smoke_events = 0
            
            # Count by type
            for b in behaviors:
                if b.behavior_type == 'loitering':
                    stats.loitering_events += 1
                elif b.behavior_type == 'running':
                    stats.running_events += 1
                elif b.behavior_type == 'violence':
                    stats.violence_events += 1
                elif b.behavior_type == 'fallen':
                    stats.fallen_events += 1
                elif b.behavior_type == 'crowd':
                    stats.crowd_events += 1
                elif b.behavior_type == 'fire':
                    stats.fire_events += 1
                elif b.behavior_type == 'smoke':
                    stats.smoke_events += 1
            
            # Count geo-fence stats
            stats.persons_in_fence = sum(1 for d in detections if d.in_geo_fence)
            stats.persons_outside_fence = sum(1 for d in detections if not d.in_geo_fence)
            
            stats.updated_at = datetime.utcnow()
            
            return stats.to_dict()
    
    def get_hourly_stats(self, camera_id, start_time=None, end_time=None):
        """
        Get hourly statistics for a camera
        
        Args:
            camera_id: Camera identifier
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            List of CameraStatistics dicts
        """
        with self.session_scope() as session:
            query = session.query(CameraStatistics).filter(
                CameraStatistics.camera_id == camera_id
            )
            
            if start_time:
                query = query.filter(CameraStatistics.hour >= start_time)
            if end_time:
                query = query.filter(CameraStatistics.hour <= end_time)
            
            query = query.order_by(CameraStatistics.hour.asc())
            
            return [s.to_dict() for s in query.all()]
    
    # ============================================================================
    # PERSON JOURNEY OPERATIONS
    # ============================================================================
    
    def update_person_journey(self, persistent_id):
        """
        Update or create a person's journey record
        
        Args:
            persistent_id: Person's UID
        
        Returns:
            PersonJourney dict
        """
        with self.session_scope() as session:
            # Get or create journey
            journey = session.query(PersonJourney).filter(
                PersonJourney.persistent_id == persistent_id,
                PersonJourney.is_active == True
            ).first()
            
            if not journey:
                # Get first detection
                first_detection = session.query(PersonDetection).filter(
                    PersonDetection.persistent_id == persistent_id
                ).order_by(PersonDetection.timestamp.asc()).first()
                
                if not first_detection:
                    return None
                
                journey = PersonJourney(
                    persistent_id=persistent_id,
                    first_seen=first_detection.timestamp,
                    last_seen=first_detection.timestamp,
                    cameras_visited=[first_detection.camera_id],
                    total_duration_seconds=0
                )
                session.add(journey)
            
            # Update with latest data
            detections = session.query(PersonDetection).filter(
                PersonDetection.persistent_id == persistent_id
            ).order_by(PersonDetection.timestamp.asc()).all()
            
            if detections:
                journey.last_seen = detections[-1].timestamp
                journey.cameras_visited = list(set(d.camera_id for d in detections))
                journey.total_duration_seconds = int((journey.last_seen - journey.first_seen).total_seconds())
                
                # Count time in/out of fence
                journey.time_in_fence_seconds = sum(
                    1 for d in detections if d.in_geo_fence
                )
                journey.time_outside_fence_seconds = len(detections) - journey.time_in_fence_seconds
            
            # Count behaviors
            behaviors = session.query(BehaviorEvent).join(PersonDetection).filter(
                PersonDetection.persistent_id == persistent_id
            ).all()
            
            journey.total_behaviors = len(behaviors)
            journey.behavior_types = list(set(b.behavior_type for b in behaviors))
            
            journey.updated_at = datetime.utcnow()
            
            return journey.to_dict()
    
    def get_person_journey(self, persistent_id):
        """
        Get a person's journey
        
        Args:
            persistent_id: Person's UID
        
        Returns:
            PersonJourney dict or None
        """
        with self.session_scope() as session:
            journey = session.query(PersonJourney).filter(
                PersonJourney.persistent_id == persistent_id
            ).order_by(PersonJourney.created_at.desc()).first()
            
            return journey.to_dict() if journey else None
    
    def close_person_journey(self, persistent_id):
        """
        Mark a person's journey as inactive
        
        Args:
            persistent_id: Person's UID
        """
        with self.session_scope() as session:
            journey = session.query(PersonJourney).filter(
                PersonJourney.persistent_id == persistent_id,
                PersonJourney.is_active == True
            ).first()
            
            if journey:
                journey.is_active = False
                journey.updated_at = datetime.utcnow()
    
    # ============================================================================
    # DAILY REPORT OPERATIONS
    # ============================================================================
    
    def generate_daily_report(self, camera_id, date):
        """
        Generate or update daily report for a camera
        Supports new behavior types (violence, fallen, fire, smoke)
        
        Args:
            camera_id: Camera identifier
            date: Date to generate report for
        
        Returns:
            DailyReport dict
        """
        with self.session_scope() as session:
            # Round to day
            day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            # Get or create report
            report = session.query(DailyReport).filter(
                DailyReport.camera_id == camera_id,
                DailyReport.date == day_start
            ).first()
            
            if not report:
                report = DailyReport(camera_id=camera_id, date=day_start)
                session.add(report)
            
            # Get all detections for the day
            detections = session.query(PersonDetection).filter(
                PersonDetection.camera_id == camera_id,
                PersonDetection.timestamp >= day_start,
                PersonDetection.timestamp < day_end
            ).all()
            
            report.total_detections = len(detections)
            report.total_unique_persons = len(set(d.persistent_id for d in detections))
            
            # Calculate hourly average
            if detections:
                report.avg_persons_per_hour = report.total_unique_persons / 24.0
            
            # Find peak hour
            if detections:
                from collections import Counter
                hours = [d.timestamp.hour for d in detections]
                hour_counts = Counter(hours)
                peak_hour, peak_count = hour_counts.most_common(1)[0]
                report.peak_hour = peak_hour
                report.peak_hour_count = peak_count
            
            # Get all behaviors for the day
            behaviors = session.query(BehaviorEvent).join(PersonDetection).filter(
                PersonDetection.camera_id == camera_id,
                BehaviorEvent.timestamp >= day_start,
                BehaviorEvent.timestamp < day_end
            ).all()
            
            # Reset counts
            report.total_loitering = 0
            report.total_running = 0
            report.total_violence = 0
            report.total_fallen = 0
            report.total_crowd = 0
            report.total_fire = 0
            report.total_smoke = 0
            
            # Count by type
            for b in behaviors:
                if b.behavior_type == 'loitering':
                    report.total_loitering += 1
                elif b.behavior_type == 'running':
                    report.total_running += 1
                elif b.behavior_type == 'violence':
                    report.total_violence += 1
                elif b.behavior_type == 'fallen':
                    report.total_fallen += 1
                elif b.behavior_type == 'crowd':
                    report.total_crowd += 1
                elif b.behavior_type == 'fire':
                    report.total_fire += 1
                elif b.behavior_type == 'smoke':
                    report.total_smoke += 1
            
            # Geo-fence averages
            if detections:
                report.avg_persons_in_fence = sum(1 for d in detections if d.in_geo_fence) / len(detections)
                report.avg_persons_outside_fence = sum(1 for d in detections if not d.in_geo_fence) / len(detections)
            
            report.updated_at = datetime.utcnow()
            
            return report.to_dict()
    
    def get_daily_reports(self, camera_id, start_date=None, end_date=None):
        """
        Get daily reports for a camera
        
        Args:
            camera_id: Camera identifier
            start_date: Start date
            end_date: End date
        
        Returns:
            List of DailyReport dicts
        """
        with self.session_scope() as session:
            query = session.query(DailyReport).filter(
                DailyReport.camera_id == camera_id
            )
            
            if start_date:
                query = query.filter(DailyReport.date >= start_date)
            if end_date:
                query = query.filter(DailyReport.date <= end_date)
            
            query = query.order_by(DailyReport.date.asc())
            
            return [r.to_dict() for r in query.all()]
    
    # ============================================================================
    # CLEANUP OPERATIONS
    # ============================================================================
    
    def cleanup_old_data(self, days_to_keep=30):
        """
        Remove data older than specified days
        
        Args:
            days_to_keep: Number of days to keep
        
        Returns:
            Dict with counts of deleted records
        """
        with self.session_scope() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old detections (cascades to behaviors)
            detections_deleted = session.query(PersonDetection).filter(
                PersonDetection.timestamp < cutoff_date
            ).delete()
            
            # Delete old stats
            stats_deleted = session.query(CameraStatistics).filter(
                CameraStatistics.hour < cutoff_date
            ).delete()
            
            # Delete old journeys
            journeys_deleted = session.query(PersonJourney).filter(
                PersonJourney.updated_at < cutoff_date,
                PersonJourney.is_active == False
            ).delete()
            
            return {
                'detections_deleted': detections_deleted,
                'stats_deleted': stats_deleted,
                'journeys_deleted': journeys_deleted
            }


# ============================================================================
# INITIALIZATION FUNCTION FOR FLASK APP
# ============================================================================

def init_db(app=None):
    """
    Initialize the database for the Flask application
    
    Args:
        app: Flask application instance (optional)
    
    Returns:
        AnalyticsDatabase instance
    """
    db_path = os.getenv('DATABASE_PATH', 'analytics.db')
    
    if app:
        # Get database path from Flask config if available
        db_path = app.config.get('DATABASE_PATH', db_path)
        print(f"[Flask] Initializing database from Flask app config")
    
    global analytics_db
    analytics_db = AnalyticsDatabase(db_path)
    
    return analytics_db


# Global instance
analytics_db = None

# Auto-initialize with default path
try:
    analytics_db = AnalyticsDatabase()
except Exception as e:
    print(f"[AnalyticsDB] ⚠️ Could not auto-initialize database: {e}")
    print("[AnalyticsDB] ℹ️ Call init_db() to initialize manually")
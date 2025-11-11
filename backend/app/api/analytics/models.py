# app/api/analytics/models.py
"""
SQLAlchemy models for analytics database
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class PersonDetection(Base):
    """
    Stores individual person detections from video frames
    """
    __tablename__ = 'person_detections'
    
    id = Column(Integer, primary_key=True)
    persistent_id = Column(Integer, nullable=False, index=True)  # Person UID
    camera_id = Column(String(50), nullable=False, index=True)
    track_id = Column(Integer, nullable=False)
    
    # Bounding box
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_width = Column(Float, nullable=False)
    bbox_height = Column(Float, nullable=False)
    
    # Detection metadata
    confidence = Column(Float, nullable=False)
    frame_id = Column(Integer, default=0)
    
    # Geo-fence information
    in_geo_fence = Column(Boolean, default=False, index=True)
    fence_id = Column(String(50), nullable=True)
    fence_name = Column(String(100), nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Relationships
    behaviors = relationship('BehaviorEvent', back_populates='detection', cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'persistent_id': self.persistent_id,
            'camera_id': self.camera_id,
            'track_id': self.track_id,
            'bbox': {
                'x': self.bbox_x,
                'y': self.bbox_y,
                'width': self.bbox_width,
                'height': self.bbox_height
            },
            'confidence': self.confidence,
            'frame_id': self.frame_id,
            'in_geo_fence': self.in_geo_fence,
            'fence_id': self.fence_id,
            'fence_name': self.fence_name,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class BehaviorEvent(Base):
    """
    Stores detected behavior events (loitering, running, suspicious, crowd)
    """
    __tablename__ = 'behavior_events'
    
    id = Column(Integer, primary_key=True)
    detection_id = Column(Integer, ForeignKey('person_detections.id'), nullable=False, index=True)
    
    # Behavior information
    behavior_type = Column(String(50), nullable=False, index=True)  # loitering, running, suspicious, crowd
    severity = Column(String(20), nullable=False, index=True)  # low, medium, high
    confidence = Column(Float, nullable=False)
    description = Column(String(500), nullable=True)
    
    # Position where behavior occurred
    position_x = Column(Float, nullable=True)
    position_y = Column(Float, nullable=True)
    
    # Additional metadata (behavior-specific data)
    meta_data = Column(JSON, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Relationships
    detection = relationship('PersonDetection', back_populates='behaviors')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'detection_id': self.detection_id,
            'behavior_type': self.behavior_type,
            'severity': self.severity,
            'confidence': self.confidence,
            'description': self.description,
            'position': {
                'x': self.position_x,
                'y': self.position_y
            } if self.position_x is not None else None,
            'metadata': self.meta_data,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class CameraStatistics(Base):
    """
    Stores hourly statistics for each camera
    """
    __tablename__ = 'camera_statistics'
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(String(50), nullable=False, index=True)
    hour = Column(DateTime, nullable=False, index=True)  # Rounded to hour
    
    # Detection statistics
    total_detections = Column(Integer, default=0)
    unique_persons = Column(Integer, default=0)
    avg_persons_per_frame = Column(Float, default=0.0)
    peak_persons = Column(Integer, default=0)
    
    # Behavior counts
    loitering_events = Column(Integer, default=0)
    running_events = Column(Integer, default=0)
    suspicious_events = Column(Integer, default=0)
    crowd_events = Column(Integer, default=0)
    
    # Geo-fence statistics
    persons_in_fence = Column(Integer, default=0)
    persons_outside_fence = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'hour': self.hour.isoformat() if self.hour else None,
            'total_detections': self.total_detections,
            'unique_persons': self.unique_persons,
            'avg_persons_per_frame': self.avg_persons_per_frame,
            'peak_persons': self.peak_persons,
            'loitering_events': self.loitering_events,
            'running_events': self.running_events,
            'suspicious_events': self.suspicious_events,
            'crowd_events': self.crowd_events,
            'persons_in_fence': self.persons_in_fence,
            'persons_outside_fence': self.persons_outside_fence,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class PersonJourney(Base):
    """
    Tracks a person's journey across cameras and time
    """
    __tablename__ = 'person_journeys'
    
    id = Column(Integer, primary_key=True)
    persistent_id = Column(Integer, nullable=False, index=True)
    
    # Journey timeline
    first_seen = Column(DateTime, nullable=False, index=True)
    last_seen = Column(DateTime, nullable=False)
    total_duration_seconds = Column(Integer, default=0)
    
    # Cameras visited
    cameras_visited = Column(JSON, default=list)  # List of camera IDs
    
    # Behavior summary
    total_behaviors = Column(Integer, default=0)
    behavior_types = Column(JSON, default=list)  # List of behavior types detected
    
    # Geo-fence time
    time_in_fence_seconds = Column(Integer, default=0)
    time_outside_fence_seconds = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'persistent_id': self.persistent_id,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'total_duration_seconds': self.total_duration_seconds,
            'cameras_visited': self.cameras_visited or [],
            'total_behaviors': self.total_behaviors,
            'behavior_types': self.behavior_types or [],
            'time_in_fence_seconds': self.time_in_fence_seconds,
            'time_outside_fence_seconds': self.time_outside_fence_seconds,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class DailyReport(Base):
    """
    Stores daily aggregated reports for each camera
    """
    __tablename__ = 'daily_reports'
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(String(50), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)  # Date (midnight)
    
    # Overall statistics
    total_detections = Column(Integer, default=0)
    total_unique_persons = Column(Integer, default=0)
    avg_persons_per_hour = Column(Float, default=0.0)
    
    # Peak information
    peak_hour = Column(Integer, nullable=True)  # Hour of day (0-23)
    peak_hour_count = Column(Integer, default=0)
    
    # Behavior totals
    total_loitering = Column(Integer, default=0)
    total_running = Column(Integer, default=0)
    total_suspicious = Column(Integer, default=0)
    total_crowd = Column(Integer, default=0)
    
    # Geo-fence averages
    avg_persons_in_fence = Column(Float, default=0.0)
    avg_persons_outside_fence = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'date': self.date.isoformat() if self.date else None,
            'total_detections': self.total_detections,
            'total_unique_persons': self.total_unique_persons,
            'avg_persons_per_hour': self.avg_persons_per_hour,
            'peak_hour': self.peak_hour,
            'peak_hour_count': self.peak_hour_count,
            'total_loitering': self.total_loitering,
            'total_running': self.total_running,
            'total_suspicious': self.total_suspicious,
            'total_crowd': self.total_crowd,
            'avg_persons_in_fence': self.avg_persons_in_fence,
            'avg_persons_outside_fence': self.avg_persons_outside_fence,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
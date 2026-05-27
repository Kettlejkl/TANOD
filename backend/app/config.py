# app/api/analytics/config.py
"""
Configuration settings for analytics module
Centralized configuration for easy customization
"""

import os
from pathlib import Path

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# SQLite database path
DB_PATH = os.getenv('ANALYTICS_DB_PATH', 'analytics.db')

# Enable/disable database operations
DB_ENABLED = os.getenv('ANALYTICS_DB_ENABLED', 'true').lower() == 'true'

# Data retention (days)
DATA_RETENTION_DAYS = int(os.getenv('ANALYTICS_RETENTION_DAYS', '30'))

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

# CSV export directory
EXPORT_DIR = os.getenv('ANALYTICS_EXPORT_DIR', 'exports')

# Create export directory if it doesn't exist
Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)

# Export format settings
EXPORT_SETTINGS = {
    'csv_delimiter': ',',
    'csv_encoding': 'utf-8',
    'include_headers': True,
    'timestamp_format': '%Y-%m-%d %H:%M:%S',
    'float_precision': 3
}

# ============================================================================
# SCHEDULER CONFIGURATION
# ============================================================================

# Enable/disable automatic exports
SCHEDULER_ENABLED = os.getenv('ANALYTICS_SCHEDULER_ENABLED', 'true').lower() == 'true'

# Export interval (seconds)
# Default: 3600 (1 hour)
# For testing: 600 (10 minutes)
SCHEDULER_INTERVAL = int(os.getenv('ANALYTICS_SCHEDULER_INTERVAL', '3600'))

# Camera IDs to monitor (comma-separated)
CAMERA_IDS_ENV = os.getenv('ANALYTICS_CAMERA_IDS', 'CAM001,CAM002')
CAMERA_IDS = [cam.strip() for cam in CAMERA_IDS_ENV.split(',') if cam.strip()]

# Export time window (hours to look back)
EXPORT_WINDOW_HOURS = int(os.getenv('ANALYTICS_EXPORT_WINDOW_HOURS', '1'))

# ============================================================================
# AGGREGATION SETTINGS
# ============================================================================

# Person-level aggregation thresholds
AGGREGATION = {
    # Minimum detections to include a person in summary
    'min_detections_per_person': 5,
    
    # Minimum duration (seconds) to include a person
    'min_duration_seconds': 10,
    
    # Behavior count thresholds for reporting
    'significant_behavior_count': 3,
    
    # Confidence threshold for including detections
    'min_confidence': 0.7
}

# ============================================================================
# BEHAVIOR DETECTION THRESHOLDS
# ============================================================================

# These match behavior_detector.py defaults but can be overridden here
# UPDATED: New detection methods (violence=YOLO-Pose, fire/smoke=YOLO)
BEHAVIOR_THRESHOLDS = {
    'loitering': {
        'time_threshold': 45.0,  # seconds
        'distance_threshold': 150,  # pixels
        'velocity_threshold': 3.0  # px/s
    },
    'running': {
        'velocity_threshold': 25.0,  # px/s
        'duration_threshold': 2.0  # seconds
    },
    'violence': {
        # YOLO-Pose based detection
        'arm_movement_threshold': 40.0,  # px/s
        'proximity_threshold': 100,  # pixels
        'pose_variance_threshold': 150.0,
        'duration_threshold': 1.0  # seconds
    },
    'fallen': {
        'standing_aspect_ratio': 1.5,  # height/width when standing
        'fallen_aspect_ratio': 0.7,  # height/width when fallen
        'y_change_threshold': 80,  # pixels (vertical drop)
        'duration_threshold': 3.0,  # seconds
        'velocity_threshold': 2.0  # px/s (must be stationary)
    },
    'fire': {
        # YOLO object detection based
        'confidence_threshold': 0.5,
        'persistence_threshold': 3.0,  # seconds
        'area_threshold': 0.03  # % of frame (color fallback)
    },
    'smoke': {
        # YOLO object detection based
        'confidence_threshold': 0.4,
        'persistence_threshold': 3.0,  # seconds
        'area_threshold': 0.05  # % of frame (color fallback)
    },
    'crowd': {
        'density_threshold': 12,  # persons
        'area_size': 200,  # pixels
        'high_density_threshold': 20  # persons
    }
}

# Valid behavior types (updated list)
VALID_BEHAVIOR_TYPES = [
    'loitering',
    'running',
    'violence',  # NEW: YOLO-Pose based
    'fallen',    # NEW: Movement + aspect ratio
    'crowd',
    'fire',      # NEW: YOLO object detection
    'smoke'      # NEW: YOLO object detection
]

# Severity mapping for behaviors
BEHAVIOR_SEVERITY_MAP = {
    'loitering': 'medium',
    'running': 'high',
    'violence': 'high',
    'fallen': 'high',
    'crowd': 'medium',
    'fire': 'critical',
    'smoke': 'high'
}

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Default limits for API queries
API_DEFAULTS = {
    'max_detections': 10000,
    'max_behaviors': 5000,
    'max_journeys': 1000,
    'default_limit': 100
}

# Rate limiting (requests per minute)
API_RATE_LIMITS = {
    'detections': 60,
    'behaviors': 60,
    'export': 10,
    'dashboard': 120
}

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

# Dashboard data refresh intervals (seconds)
DASHBOARD_REFRESH = {
    'summary': 5,  # Header metrics
    'analytics': 30,  # Charts and graphs
    'anomalies': 10,  # Recent alerts
    'roi': 15,  # ROI occupancy
    'incidents': 5  # Active incidents
}

# Dashboard data time windows
DASHBOARD_WINDOWS = {
    'recent_anomalies': 60,  # minutes
    'incident_lookback': 60,  # minutes
    'occupancy_window': 5,  # minutes
    'trend_hours': 12  # hours
}

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# Database query optimization
DB_OPTIMIZATION = {
    'batch_size': 1000,  # Batch insert size
    'connection_pool_size': 5,
    'query_timeout': 30,  # seconds
    'enable_wal_mode': True  # SQLite WAL mode
}

# Export performance
EXPORT_OPTIMIZATION = {
    'chunk_size': 10000,  # Rows per chunk
    'use_compression': False,  # Gzip compression
    'parallel_exports': False  # Parallel export (experimental)
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log levels
LOG_LEVEL = os.getenv('ANALYTICS_LOG_LEVEL', 'INFO')

# Enable detailed logging
DEBUG_MODE = os.getenv('ANALYTICS_DEBUG', 'false').lower() == 'true'

# Log file path
LOG_FILE = os.getenv('ANALYTICS_LOG_FILE', 'analytics.log')

# ============================================================================
# INTEGRATION SETTINGS
# ============================================================================

# Integration with stream_manager
STREAM_INTEGRATION = {
    'db_save_interval': 5,  # frames between DB saves
    'behavior_analysis_interval': 3,  # frames between behavior checks
    'max_age_inactive_tracks': 30.0  # seconds
}

# Integration with alerts system
ALERTS_INTEGRATION = {
    'send_alerts': True,
    'alerts_api_url': 'http://127.0.0.1:5000/api/alerts',
    'alerts_timeout': 2.0,  # seconds
    'max_queue_size': 100
}

# ============================================================================
# FEATURE FLAGS
# ============================================================================

FEATURES = {
    'enable_person_journeys': True,
    'enable_cross_camera_tracking': True,
    'enable_behavior_detection': True,
    'enable_hourly_stats': True,
    'enable_daily_reports': True,
    'enable_scheduler': SCHEDULER_ENABLED,
    'enable_api_endpoints': True
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required directories
    if not os.path.exists(EXPORT_DIR):
        try:
            Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create export directory: {e}")
    
    # Check scheduler settings
    if SCHEDULER_ENABLED and not CAMERA_IDS:
        errors.append("Scheduler enabled but no camera IDs configured")
    
    if SCHEDULER_INTERVAL < 60:
        errors.append(f"Warning: Scheduler interval very short ({SCHEDULER_INTERVAL}s)")
    
    # Check database settings
    if DB_ENABLED and not DB_PATH:
        errors.append("Database enabled but no path specified")
    
    if errors:
        print("[Analytics Config] ⚠️  Configuration warnings:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("[Analytics Config] ✅ Configuration validated")
    return True


def print_config():
    """Print current configuration"""
    print("\n" + "="*80)
    print("ANALYTICS CONFIGURATION")
    print("="*80)
    print(f"Database: {DB_PATH} ({'enabled' if DB_ENABLED else 'disabled'})")
    print(f"Export Directory: {EXPORT_DIR}")
    print(f"Scheduler: {'enabled' if SCHEDULER_ENABLED else 'disabled'}")
    if SCHEDULER_ENABLED:
        print(f"  Interval: {SCHEDULER_INTERVAL}s ({SCHEDULER_INTERVAL/3600:.1f}h)")
        print(f"  Cameras: {', '.join(CAMERA_IDS)}")
    print(f"Data Retention: {DATA_RETENTION_DAYS} days")
    print(f"Debug Mode: {'enabled' if DEBUG_MODE else 'disabled'}")
    print("="*80 + "\n")


# Run validation on import
if __name__ != "__main__":
    validate_config()
    if DEBUG_MODE:
        print_config()
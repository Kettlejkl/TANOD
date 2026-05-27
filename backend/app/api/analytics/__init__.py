# app/api/analytics/__init__.py
"""
Analytics Module - Person Tracking & Behavior Analysis
Provides database storage, CSV export, and API endpoints for surveillance analytics
"""

from .database import analytics_db, AnalyticsDatabase
from .exporter import exporter, AnalyticsExporter
from .routes import analytics_bp
from .scheduler import (
    start_scheduler, 
    stop_scheduler, 
    force_export,
    get_scheduler_status,
    start_scheduler_thread  # Legacy name
)

__all__ = [
    # Database
    'analytics_db',
    'AnalyticsDatabase',
    
    # Exporter
    'exporter',
    'AnalyticsExporter',
    
    # Routes
    'analytics_bp',
    
    # Scheduler functions
    'start_scheduler',
    'stop_scheduler',
    'force_export',
    'get_scheduler_status',
    'start_scheduler_thread',  # Legacy
]

__version__ = '1.0.0'
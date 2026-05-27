"""
ReID Package - Modular Person Re-Identification and Tracking System

This package provides a complete solution for person tracking across multiple cameras
using deep learning-based re-identification, appearance analysis, and motion tracking.

Main Components:
- FeatureExtractor: ReID feature extraction using TorchREID models
- AppearanceAnalyzer: Color histogram and clothing attribute analysis
- MotionTracker: Kalman filtering and trajectory tracking
- FeatureMatcher: Similarity computation and matching logic
- PersonDatabase: Identity and metadata management
- TrackingManager: Main orchestrator combining all components
"""

from .feature_extractor import FeatureExtractor
from .appearance_analyzer import AppearanceAnalyzer
from .motion_tracker import MotionTracker, KalmanFilter
from .feature_matcher import FeatureMatcher
from .person_database import PersonDatabase
from .tracking_manager import TrackingManager

__version__ = '1.0.0'
__author__ = 'Your Name'

__all__ = [
    'FeatureExtractor',
    'AppearanceAnalyzer',
    'MotionTracker',
    'KalmanFilter',
    'FeatureMatcher',
    'PersonDatabase',
    'TrackingManager',
]


def create_tracker(config=None):
    """
    Convenience function to create a fully configured tracking manager.
    
    Args:
        config (dict, optional): Configuration dictionary with component settings
        
    Returns:
        TrackingManager: Configured tracking manager instance
        
    Example:
        >>> from reid import create_tracker
        >>> tracker = create_tracker()
        >>> results = tracker.process_detections(camera_id, detections, frame)
    """
    if config is None:
        config = {}
    
    # Extract component configs
    feature_config = config.get('feature_extractor', {})
    appearance_config = config.get('appearance_analyzer', {})
    motion_config = config.get('motion_tracker', {})
    matcher_config = config.get('feature_matcher', {})
    database_config = config.get('person_database', {})
    manager_config = config.get('tracking_manager', {})
    
    # Create components
    feature_extractor = FeatureExtractor(**feature_config)
    appearance_analyzer = AppearanceAnalyzer(**appearance_config)
    motion_tracker = MotionTracker(**motion_config)
    feature_matcher = FeatureMatcher(**matcher_config)
    person_database = PersonDatabase(**database_config)
    
    # Create manager
    tracking_manager = TrackingManager(
        feature_extractor=feature_extractor,
        appearance_analyzer=appearance_analyzer,
        motion_tracker=motion_tracker,
        feature_matcher=feature_matcher,
        person_database=person_database,
        **manager_config
    )
    
    return tracking_manager


# Preset configurations
PRESET_CONFIGS = {
    'default': {
        'feature_matcher': {
            'similarity_threshold': 0.35,
            'cross_camera_threshold': 0.45,
        },
        'person_database': {
            'max_features_per_person': 200,
            'max_tracked_persons': 1000,
        }
    },
    
    'high_accuracy': {
        'feature_matcher': {
            'similarity_threshold': 0.45,
            'cross_camera_threshold': 0.55,
            'spatial_proximity_threshold': 80,
            'color_weight': 0.40,
            'clothing_weight': 0.40,
        },
        'person_database': {
            'max_features_per_person': 300,
        }
    },
    
    'fast': {
        'feature_matcher': {
            'similarity_threshold': 0.30,
            'cross_camera_threshold': 0.40,
            'spatial_proximity_threshold': 150,
            'color_weight': 0.20,
            'clothing_weight': 0.20,
        },
        'person_database': {
            'max_features_per_person': 100,
            'max_tracked_persons': 500,
        }
    },
    
    'memory_efficient': {
        'person_database': {
            'max_features_per_person': 50,
            'max_tracked_persons': 500,
            'max_age': 180.0,
        }
    }
}


def create_tracker_preset(preset='default'):
    """
    Create a tracker with a preset configuration.
    
    Args:
        preset (str): Preset name - 'default', 'high_accuracy', 'fast', or 'memory_efficient'
        
    Returns:
        TrackingManager: Configured tracking manager instance
        
    Example:
        >>> from reid import create_tracker_preset
        >>> tracker = create_tracker_preset('high_accuracy')
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    
    return create_tracker(PRESET_CONFIGS[preset])
# app/api/tracker_debug_routes.py
# Debug API endpoints for monitoring UID tracking stability

from flask import Blueprint, jsonify, request
from app.api.video.stream_manager import stream_manager

tracker_debug_bp = Blueprint('tracker_debug', __name__)


@tracker_debug_bp.route('/api/tracker/stats', methods=['GET'])
def get_global_tracker_stats():
    """
    Get global tracker statistics
    
    Returns:
        - total_persistent_ids: Total unique UIDs created
        - active_tracks: Currently tracked objects
        - pending_cross_matches: Cross-camera matches being confirmed
        - next_id: Next UID that will be assigned
        - active_uids_total: Total active UIDs across all cameras
        - active_per_camera: Active UIDs per camera
        - feature_failures: Total tracks with feature extraction issues
    """
    try:
        stats = stream_manager.get_tracking_statistics()
        if stats:
            return jsonify({
                'success': True,
                'stats': stats
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': 'Tracker not initialized'
            }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@tracker_debug_bp.route('/api/tracker/debug/<camera_id>', methods=['GET'])
def get_camera_debug_info(camera_id):
    """
    Get detailed debug information for a specific camera
    
    Args:
        camera_id: Camera identifier
        
    Returns:
        - global_stats: Overall tracker statistics
        - camera_specific:
            - active_uids: List of active UIDs in this camera
            - pending_matches: Cross-camera matches being confirmed
            - feature_failures: Tracks with feature extraction issues
    """
    try:
        debug_info = stream_manager.get_tracker_debug_info(camera_id)
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'debug_info': debug_info
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@tracker_debug_bp.route('/api/tracker/uid-history/<int:persistent_id>', methods=['GET'])
def get_uid_history(persistent_id):
    """
    Get tracking history for a specific UID
    
    Args:
        persistent_id: Persistent ID to query
        
    Returns:
        - camera_history: List of cameras where person was seen
        - first_seen: When UID was first created
        - last_seen: Most recent detection
        - feature_count: Number of stored features
        - geo_fence_entry: When entered geo-fence (if applicable)
    """
    try:
        tracker = stream_manager.persistent_tracker
        
        if persistent_id not in tracker.persistent_ids:
            return jsonify({
                'success': False,
                'message': f'UID {persistent_id} not found'
            }), 404
        
        history = {
            'persistent_id': persistent_id,
            'camera_history': list(tracker.camera_history.get(persistent_id, set())),
            'current_camera': tracker.camera_locations.get(persistent_id),
            'first_seen': tracker.first_seen.get(persistent_id),
            'last_seen': tracker.last_seen.get(persistent_id),
            'feature_count': len(tracker.feature_history.get(persistent_id, [])),
            'geo_fence_entry': tracker.geo_fence_entry.get(persistent_id),
            'spatial_context': tracker.spatial_context.get(persistent_id)
        }
        
        return jsonify({
            'success': True,
            'history': history
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@tracker_debug_bp.route('/api/tracker/active-uids/<camera_id>', methods=['GET'])
def get_active_uids(camera_id):
    """
    Get list of currently active UIDs in a camera
    
    Args:
        camera_id: Camera identifier
        
    Returns:
        - active_uids: List of active persistent IDs
        - count: Number of active UIDs
    """
    try:
        active_uids = stream_manager.persistent_tracker.get_active_uids_in_camera(camera_id)
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'active_uids': list(active_uids),
            'count': len(active_uids)
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@tracker_debug_bp.route('/api/tracker/pending-matches', methods=['GET'])
def get_all_pending_matches():
    """
    Get all pending cross-camera matches across all cameras
    
    Returns:
        - pending_matches: List of all pending cross-camera confirmations
    """
    try:
        tracker = stream_manager.persistent_tracker
        pending_list = []
        
        for (cam_id, track_id), pending in tracker.pending_cross_matches.items():
            import numpy as np
            pending_list.append({
                'camera_id': cam_id,
                'track_id': track_id,
                'target_pid': pending['pid'],
                'progress': f"{pending['count']}/{tracker.confirmation_frames}",
                'avg_confidence': float(np.mean(pending['scores'])),
                'min_confidence': float(min(pending['scores'])),
                'max_confidence': float(max(pending['scores'])),
                'failures': pending.get('failures', 0),
                'first_seen': pending['first_seen']
            })
        
        return jsonify({
            'success': True,
            'pending_matches': pending_list,
            'total': len(pending_list)
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@tracker_debug_bp.route('/api/tracker/feature-failures', methods=['GET'])
def get_feature_failures():
    """
    Get all tracks experiencing feature extraction failures
    
    Returns:
        - failures: List of tracks with feature extraction issues
    """
    try:
        tracker = stream_manager.persistent_tracker
        failures_list = []
        
        for (cam_id, track_id), failure_count in tracker.feature_extraction_failures.items():
            if failure_count > 0:
                failures_list.append({
                    'camera_id': cam_id,
                    'track_id': track_id,
                    'failure_count': failure_count,
                    'max_failures': tracker.feature_failure_patience,
                    'progress': f"{failure_count}/{tracker.feature_failure_patience}"
                })
        
        return jsonify({
            'success': True,
            'feature_failures': failures_list,
            'total': len(failures_list)
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@tracker_debug_bp.route('/api/tracker/similarity-test', methods=['POST'])
def test_similarity():
    """
    Test similarity calculation between two feature vectors
    
    Request body:
        - pid1: First persistent ID
        - pid2: Second persistent ID
        
    Returns:
        - similarity: Cosine similarity score (0-1)
    """
    try:
        data = request.json
        pid1 = data.get('pid1')
        pid2 = data.get('pid2')
        
        if not pid1 or not pid2:
            return jsonify({
                'success': False,
                'message': 'Both pid1 and pid2 required'
            }), 400
        
        tracker = stream_manager.persistent_tracker
        
        if pid1 not in tracker.persistent_ids or pid2 not in tracker.persistent_ids:
            return jsonify({
                'success': False,
                'message': 'One or both PIDs not found'
            }), 404
        
        feature1 = tracker.persistent_ids[pid1]
        feature2 = tracker.persistent_ids[pid2]
        
        similarity = tracker._calculate_similarity(feature1, feature2)
        
        return jsonify({
            'success': True,
            'pid1': pid1,
            'pid2': pid2,
            'similarity': float(similarity),
            'would_match_same_camera': similarity >= tracker.similarity_threshold,
            'would_match_cross_camera': similarity >= tracker.cross_camera_threshold,
            'thresholds': {
                'same_camera': tracker.similarity_threshold,
                'cross_camera': tracker.cross_camera_threshold
            }
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@tracker_debug_bp.route('/api/tracker/reset-uid/<int:persistent_id>', methods=['POST'])
def reset_uid(persistent_id):
    """
    Reset/remove a specific UID (for debugging/testing only)
    
    Args:
        persistent_id: UID to reset
        
    Returns:
        - success: Whether operation succeeded
    """
    try:
        tracker = stream_manager.persistent_tracker
        
        if persistent_id not in tracker.persistent_ids:
            return jsonify({
                'success': False,
                'message': f'UID {persistent_id} not found'
            }), 404
        
        # Remove from all tracking structures
        tracker.persistent_ids.pop(persistent_id, None)
        tracker.feature_history.pop(persistent_id, None)
        tracker.last_seen.pop(persistent_id, None)
        tracker.camera_locations.pop(persistent_id, None)
        tracker.spatial_context.pop(persistent_id, None)
        tracker.camera_history.pop(persistent_id, None)
        tracker.geo_fence_entry.pop(persistent_id, None)
        tracker.uid_assignment_lock.pop(persistent_id, None)
        
        # Remove from active UIDs
        for cam_id in tracker.active_uids_per_camera:
            tracker.active_uids_per_camera[cam_id].discard(persistent_id)
        
        # Remove track mappings
        tracks_to_remove = [k for k, v in tracker.track_to_persistent.items() if v == persistent_id]
        for track_key in tracks_to_remove:
            del tracker.track_to_persistent[track_key]
        
        return jsonify({
            'success': True,
            'message': f'UID {persistent_id} reset successfully'
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@tracker_debug_bp.route('/api/tracker/config', methods=['GET'])
def get_tracker_config():
    """
    Get current tracker configuration
    
    Returns:
        - Configuration parameters
    """
    try:
        tracker = stream_manager.persistent_tracker
        
        config = {
            'similarity_threshold': tracker.similarity_threshold,
            'cross_camera_threshold': tracker.cross_camera_threshold,
            'confirmation_frames': tracker.confirmation_frames,
            'cross_camera_time_window': tracker.cross_camera_time_window,
            'feature_failure_patience': tracker.feature_failure_patience,
            'min_box_area': tracker.min_box_area,
            'max_features_per_person': tracker.max_features_per_person,
            'db_enabled': tracker.db is not None,
            'use_db_for_matching': tracker.use_db_for_matching
        }
        
        return jsonify({
            'success': True,
            'config': config
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@tracker_debug_bp.route('/api/tracker/health', methods=['GET'])
def get_tracker_health():
    """
    Get tracker health status
    
    Returns:
        - Health metrics and warnings
    """
    try:
        tracker = stream_manager.persistent_tracker
        stats = tracker.get_statistics()
        
        warnings = []
        
        # Check for excessive pending matches
        if stats['pending_cross_matches'] > 10:
            warnings.append(f"High number of pending cross-camera matches: {stats['pending_cross_matches']}")
        
        # Check for feature extraction issues
        feature_failures = stats.get('feature_failures', 0)
        if feature_failures > 5:
            warnings.append(f"Multiple tracks experiencing feature extraction failures: {feature_failures}")
        
        # Check for rapid UID growth (potential duplicate UID issue)
        if stats['next_id'] > 100 and stats['active_uids_total'] < 5:
            warnings.append(f"Potential UID duplication: {stats['next_id']} UIDs created but only {stats['active_uids_total']} active")
        
        health_status = 'healthy' if len(warnings) == 0 else 'warning'
        
        return jsonify({
            'success': True,
            'status': health_status,
            'stats': stats,
            'warnings': warnings,
            'timestamp': import_time.time()
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# Import time for health endpoint
import time as import_time
# app/api/alerts.py
"""
Fixed Alert System with proper endpoints
Handles both YOLO detections and behavior alerts
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
from enum import Enum
import json
from collections import deque

# ============= CREATE BLUEPRINT =============
alerts_bp = Blueprint("alerts", __name__)

# ============= IN-MEMORY STORAGE =============
# Store alerts in memory (simple queue with max size)
alert_storage = deque(maxlen=1000)  # Keep last 1000 alerts
alert_id_counter = [0]  # Use list to make it mutable

# ============= ENUMS =============

class AlertType(Enum):
    LOITERING = "loitering"
    RUNNING = "running"
    SUSPICIOUS = "suspicious"
    CROWD = "crowd"
    GENERAL = "general"
    DETECTION = "detection"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# ============= HELPER FUNCTIONS =============

def safe_json(obj):
    """Convert object to JSON-safe format"""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj

def create_alert(alert_type, severity, location, description, metadata=None):
    """Create and store an alert"""
    alert_id_counter[0] += 1
    
    alert = {
        'id': alert_id_counter[0],
        'type': alert_type,
        'severity': severity,
        'location': location,
        'description': description,
        'metadata': safe_json(metadata) if metadata else {},
        'timestamp': datetime.now().isoformat(),
        'status': 'active'
    }
    
    alert_storage.append(alert)
    
    # Print alert to console for debugging
    severity_emoji = {
        'low': '🟢',
        'medium': '🟡',
        'high': '🔴'
    }
    print(f"{severity_emoji.get(severity, '⚪')} [ALERT] {alert_type.upper()} - {description}")
    
    return alert

# ============= API ROUTES =============

@alerts_bp.route("/yolo-detection", methods=["POST"])
def yolo_detection():
    """
    Handle YOLO detection data
    
    Expected JSON format:
    {
        "camera_id": "camera_1",
        "detections": [
            {
                "class": "person",
                "confidence": 0.95,
                "bbox": [x, y, width, height],
                "track_id": 1
            }
        ],
        "frame_id": 123,
        "timestamp": "2025-10-26T13:10:48"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        camera_id = data.get('camera_id', 'unknown')
        detections = data.get('detections', [])
        frame_id = data.get('frame_id', 0)
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        person_count = sum(1 for d in detections if d.get('class') == 'person')
        
        alerts_created = []
        
        # Create alert for high person count
        if person_count >= 10:
            alert = create_alert(
                alert_type='crowd',
                severity='high',
                location=camera_id,
                description=f'High person count detected: {person_count} people',
                metadata={
                    'person_count': person_count,
                    'frame_id': frame_id,
                    'camera_id': camera_id,
                    'timestamp': timestamp
                }
            )
            alerts_created.append(alert)
        
        return jsonify({
            "success": True,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "detections_processed": len(detections),
            "person_count": person_count,
            "alerts_created": len(alerts_created),
            "timestamp": timestamp
        }), 200
    
    except Exception as e:
        print(f"[Alerts] Error in yolo_detection: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@alerts_bp.route("/create", methods=["POST"])
def create_alert_endpoint():
    """
    Create a behavior alert
    
    Expected JSON format:
    {
        "alert_type": "behavior",
        "type": "loitering",
        "severity": "medium",
        "location": "camera_1",
        "description": "Person loitering detected",
        "metadata": {
            "person_id": 123,
            "duration": 45.5,
            "confidence": 0.85
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        alert_type = data.get('type', 'general')
        severity = data.get('severity', 'medium')
        location = data.get('location', 'unknown')
        description = data.get('description', 'Alert detected')
        metadata = data.get('metadata', {})
        
        alert = create_alert(
            alert_type=alert_type,
            severity=severity,
            location=location,
            description=description,
            metadata=metadata
        )
        
        return jsonify({
            "success": True,
            "alert_id": alert['id'],
            "alert": alert
        }), 201
    
    except Exception as e:
        print(f"[Alerts] Error in create_alert: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@alerts_bp.route("/list", methods=["GET"])
def list_alerts():
    """
    Get all alerts
    
    Query parameters:
    - limit: Maximum number of alerts to return (default: 100)
    - type: Filter by alert type
    - severity: Filter by severity
    """
    try:
        limit = int(request.args.get('limit', 100))
        alert_type_filter = request.args.get('type')
        severity_filter = request.args.get('severity')
        
        # Convert deque to list for filtering
        alerts = list(alert_storage)
        
        # Apply filters
        if alert_type_filter:
            alerts = [a for a in alerts if a['type'] == alert_type_filter]
        
        if severity_filter:
            alerts = [a for a in alerts if a['severity'] == severity_filter]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Apply limit
        alerts = alerts[:limit]
        
        return jsonify({
            "success": True,
            "count": len(alerts),
            "alerts": alerts
        }), 200
    
    except Exception as e:
        print(f"[Alerts] Error in list_alerts: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@alerts_bp.route("/clear", methods=["POST"])
def clear_alerts():
    """Clear all alerts"""
    try:
        alert_storage.clear()
        alert_id_counter[0] = 0
        
        return jsonify({
            "success": True,
            "message": "All alerts cleared"
        }), 200
    
    except Exception as e:
        print(f"[Alerts] Error in clear_alerts: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@alerts_bp.route("/stats", methods=["GET"])
def alert_stats():
    """Get alert statistics"""
    try:
        alerts = list(alert_storage)
        
        stats = {
            'total': len(alerts),
            'by_type': {},
            'by_severity': {},
            'recent_count': 0
        }
        
        # Count by type
        for alert in alerts:
            alert_type = alert['type']
            stats['by_type'][alert_type] = stats['by_type'].get(alert_type, 0) + 1
        
        # Count by severity
        for alert in alerts:
            severity = alert['severity']
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
        
        # Count recent (last 5 minutes)
        now = datetime.now()
        for alert in alerts:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if (now - alert_time).total_seconds() < 300:  # 5 minutes
                stats['recent_count'] += 1
        
        return jsonify({
            "success": True,
            "stats": stats
        }), 200
    
    except Exception as e:
        print(f"[Alerts] Error in alert_stats: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@alerts_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "success": True,
        "status": "healthy",
        "alert_count": len(alert_storage),
        "timestamp": datetime.now().isoformat()
    }), 200


# ============= EXPORTS =============
__all__ = ['alerts_bp']
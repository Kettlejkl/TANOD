# app/api/alerts.py

from flask import Blueprint, jsonify, request
from datetime import datetime
from enum import Enum
import json
from collections import deque

alerts_bp = Blueprint("alerts", __name__)

alert_storage = deque(maxlen=1000)
alert_id_counter = [0]

class AlertType(Enum):
    LOITERING = "loitering"
    RUNNING = "running"
    VIOLENCE = "violence"
    FALLEN = "fallen"
    CROWD = "crowd"
    FIRE = "fire"
    SMOKE = "smoke"
    GENERAL = "general"
    DETECTION = "detection"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj

def create_alert(alert_type, severity, location, description, metadata=None):
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
    
    severity_emoji = {
        'low': '🟢',
        'medium': '🟡',
        'high': '🔴',
        'critical': '🚨'
    }
    print(f"{severity_emoji.get(severity, '⚪')} [ALERT] {alert_type.upper()} - {description}")
    
    return alert

@alerts_bp.route("/yolo-detection", methods=["POST"])
def yolo_detection():
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
        
        valid_types = [t.value for t in AlertType]
        if alert_type not in valid_types:
            return jsonify({
                "success": False,
                "error": f"Invalid alert type. Must be one of: {valid_types}"
            }), 400
        
        valid_severities = [s.value for s in AlertSeverity]
        if severity not in valid_severities:
            return jsonify({
                "success": False,
                "error": f"Invalid severity. Must be one of: {valid_severities}"
            }), 400
        
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
    try:
        limit = int(request.args.get('limit', 100))
        alert_type_filter = request.args.get('type')
        severity_filter = request.args.get('severity')
        
        alerts = list(alert_storage)
        
        if alert_type_filter:
            alerts = [a for a in alerts if a['type'] == alert_type_filter]
        
        if severity_filter:
            alerts = [a for a in alerts if a['severity'] == severity_filter]
        
        alerts.sort(key=lambda x: x['timestamp'], reverse=True)
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
    try:
        alerts = list(alert_storage)
        
        stats = {
            'total': len(alerts),
            'by_type': {},
            'by_severity': {},
            'recent_count': 0
        }
        
        for alert in alerts:
            alert_type = alert['type']
            stats['by_type'][alert_type] = stats['by_type'].get(alert_type, 0) + 1
        
        for alert in alerts:
            severity = alert['severity']
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
        
        now = datetime.now()
        for alert in alerts:
            alert_time = datetime.fromisoformat(alert['timestamp'])
            if (now - alert_time).total_seconds() < 300:
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
    return jsonify({
        "success": True,
        "status": "healthy",
        "alert_count": len(alert_storage),
        "supported_types": [t.value for t in AlertType],
        "supported_severities": [s.value for s in AlertSeverity],
        "timestamp": datetime.now().isoformat()
    }), 200


__all__ = ['alerts_bp', 'AlertType', 'AlertSeverity', 'create_alert']
from flask import Blueprint, jsonify, request
from datetime import datetime

api = Blueprint("api", __name__)

# Temporary in-memory storage
alerts = [
    {
        "id": 1,
        "type": "overcrowding",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "active",
        "zone_ref": {"name": "Zone A - Terminal"},
        "confidence": 0.92,
        "severity": "high",
        "description": "Crowd density exceeds safe threshold",
        "camera_id": "CAM001"
    }
]

occupancy = {
    "Zone A - Terminal": 25,
    "Zone B - Bus 1 Interior": 21,
    "Zone C - Restricted Area": 2,
    "Zone D - Food Court": 9
}

# Fetch alerts
@api.route("/alerts", methods=["GET"])
def get_alerts():
    return jsonify(alerts)

# Acknowledge an alert
@api.route("/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id):
    for alert in alerts:
        if alert["id"] == alert_id:
            alert["status"] = "acknowledged"
            alert["acknowledged_by"] = "Current User"
    return jsonify({"success": True})

# Fetch occupancy
@api.route("/occupancy", methods=["GET"])
def get_occupancy():
    return jsonify(occupancy)

# Add new alert (simulated)
@api.route("/alerts", methods=["POST"])
def add_alert():
    data = request.json
    data["id"] = len(alerts) + 1
    data["timestamp"] = datetime.utcnow().isoformat()
    data["status"] = "active"
    alerts.insert(0, data)
    return jsonify(data), 201

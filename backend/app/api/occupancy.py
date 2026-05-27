from flask import Blueprint, jsonify

occupancy_bp = Blueprint("occupancy", __name__, url_prefix="/api/occupancy")

@occupancy_bp.route("/", methods=["GET"])
def get_occupancy():
    return jsonify({"message": "Occupancy API is working"})

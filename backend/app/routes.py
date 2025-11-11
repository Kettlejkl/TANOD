# app/api/analytics/routes.py
"""
Flask routes for analytics API
Provides endpoints for querying and exporting surveillance data
INCLUDES: Passenger flow trends, arrival/departure detection, peak crowding, wait times
"""

from flask import Blueprint, jsonify, request, send_file
from datetime import datetime, timedelta
from sklearn import logger
from sqlalchemy import func, case
from collections import defaultdict
import statistics
import os

from .database import analytics_db
from .exporter import exporter
from .models import PersonDetection, BehaviorEvent, PersonJourney

analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')

# ============================================================================
# DETECTION ENDPOINTS
# ============================================================================

@analytics_bp.route('/detections', methods=['GET'])
def get_detections():
    """
    Get person detections with optional filters
    
    Query Parameters:
        - camera_id: Filter by camera ID
        - persistent_id: Filter by person UID
        - start_time: Start timestamp (ISO format)
        - end_time: End timestamp (ISO format)
        - limit: Max results (default: 1000)
    
    Returns:
        JSON array of detections
    """
    try:
        camera_id = request.args.get('camera_id')
        persistent_id = request.args.get('persistent_id', type=int)
        limit = request.args.get('limit', default=1000, type=int)
        
        # Parse timestamps
        start_time = None
        end_time = None
        if request.args.get('start_time'):
            start_time = datetime.fromisoformat(request.args.get('start_time'))
        if request.args.get('end_time'):
            end_time = datetime.fromisoformat(request.args.get('end_time'))
        
        detections = analytics_db.get_detections(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            persistent_id=persistent_id,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'count': len(detections),
            'detections': detections
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/detections/export', methods=['POST'])
def export_detections():
    """
    Export detections to CSV
    
    Request Body:
        - camera_id: Camera ID (optional)
        - start_time: Start timestamp (optional)
        - end_time: End timestamp (optional)
        - filename: Output filename (optional)
    
    Returns:
        CSV file download
    """
    try:
        data = request.get_json() or {}
        
        camera_id = data.get('camera_id')
        
        start_time = None
        end_time = None
        if data.get('start_time'):
            start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            end_time = datetime.fromisoformat(data['end_time'])
        
        filename = data.get('filename')
        
        filepath = exporter.export_detections(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            filename=filename
        )
        
        if not filepath:
            return jsonify({
                'success': False,
                'error': 'No detections to export'
            }), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=os.path.basename(filepath),
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# BEHAVIOR ENDPOINTS
# ============================================================================

@analytics_bp.route('/behaviors', methods=['GET'])
def get_behaviors():
    """
    Get behavior events with optional filters
    
    Query Parameters:
        - camera_id: Filter by camera ID
        - behavior_type: Filter by type (loitering, running, suspicious, crowd)
        - severity: Filter by severity (low, medium, high)
        - start_time: Start timestamp (ISO format)
        - end_time: End timestamp (ISO format)
        - limit: Max results (default: 1000)
    
    Returns:
        JSON array of behaviors
    """
    try:
        camera_id = request.args.get('camera_id')
        behavior_type = request.args.get('behavior_type')
        severity = request.args.get('severity')
        limit = request.args.get('limit', default=1000, type=int)
        
        # Parse timestamps
        start_time = None
        end_time = None
        if request.args.get('start_time'):
            start_time = datetime.fromisoformat(request.args.get('start_time'))
        if request.args.get('end_time'):
            end_time = datetime.fromisoformat(request.args.get('end_time'))
        
        behaviors = analytics_db.get_behaviors(
            camera_id=camera_id,
            behavior_type=behavior_type,
            severity=severity,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'count': len(behaviors),
            'behaviors': behaviors
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/behaviors/export', methods=['POST'])
def export_behaviors():
    """
    Export behaviors to CSV
    
    Request Body:
        - camera_id: Camera ID (optional)
        - behavior_type: Behavior type filter (optional)
        - severity: Severity filter (optional)
        - start_time: Start timestamp (optional)
        - end_time: End timestamp (optional)
        - filename: Output filename (optional)
    
    Returns:
        CSV file download
    """
    try:
        data = request.get_json() or {}
        
        camera_id = data.get('camera_id')
        behavior_type = data.get('behavior_type')
        severity = data.get('severity')
        
        start_time = None
        end_time = None
        if data.get('start_time'):
            start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            end_time = datetime.fromisoformat(data['end_time'])
        
        filename = data.get('filename')
        
        filepath = exporter.export_behaviors(
            camera_id=camera_id,
            behavior_type=behavior_type,
            severity=severity,
            start_time=start_time,
            end_time=end_time,
            filename=filename
        )
        
        if not filepath:
            return jsonify({
                'success': False,
                'error': 'No behaviors to export'
            }), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=os.path.basename(filepath),
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# STATISTICS ENDPOINTS
# ============================================================================

@analytics_bp.route('/stats/hourly', methods=['GET'])
def get_hourly_stats():
    """
    Get hourly statistics for a camera
    
    Query Parameters:
        - camera_id: Camera identifier (required)
        - start_time: Start timestamp (ISO format, optional)
        - end_time: End timestamp (ISO format, optional)
    
    Returns:
        JSON array of hourly statistics
    """
    try:
        camera_id = request.args.get('camera_id')
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'camera_id is required'
            }), 400
        
        start_time = None
        end_time = None
        if request.args.get('start_time'):
            start_time = datetime.fromisoformat(request.args.get('start_time'))
        if request.args.get('end_time'):
            end_time = datetime.fromisoformat(request.args.get('end_time'))
        
        stats = analytics_db.get_hourly_stats(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'count': len(stats),
            'stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/stats/hourly/update', methods=['POST'])
def update_hourly_stats():
    """
    Calculate and update hourly statistics
    
    Request Body:
        - camera_id: Camera identifier (required)
        - hour: Hour timestamp (ISO format, optional - defaults to current hour)
    
    Returns:
        JSON with updated statistics
    """
    try:
        data = request.get_json() or {}
        
        camera_id = data.get('camera_id')
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'camera_id is required'
            }), 400
        
        hour_timestamp = datetime.utcnow()
        if data.get('hour'):
            hour_timestamp = datetime.fromisoformat(data['hour'])
        
        stats = analytics_db.update_hourly_stats(camera_id, hour_timestamp)
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/stats/hourly/export', methods=['POST'])
def export_hourly_stats():
    """
    Export hourly statistics to CSV
    
    Request Body:
        - camera_id: Camera identifier (required)
        - start_time: Start timestamp (optional)
        - end_time: End timestamp (optional)
        - filename: Output filename (optional)
    
    Returns:
        CSV file download
    """
    try:
        data = request.get_json() or {}
        
        camera_id = data.get('camera_id')
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'camera_id is required'
            }), 400
        
        start_time = None
        end_time = None
        if data.get('start_time'):
            start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            end_time = datetime.fromisoformat(data['end_time'])
        
        filename = data.get('filename')
        
        filepath = exporter.export_hourly_stats(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            filename=filename
        )
        
        if not filepath:
            return jsonify({
                'success': False,
                'error': 'No hourly stats to export'
            }), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=os.path.basename(filepath),
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# DAILY REPORT ENDPOINTS
# ============================================================================

@analytics_bp.route('/reports/daily', methods=['GET'])
def get_daily_reports():
    """
    Get daily reports for a camera
    
    Query Parameters:
        - camera_id: Camera identifier (required)
        - start_date: Start date (ISO format, optional)
        - end_date: End date (ISO format, optional)
    
    Returns:
        JSON array of daily reports
    """
    try:
        camera_id = request.args.get('camera_id')
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'camera_id is required'
            }), 400
        
        start_date = None
        end_date = None
        if request.args.get('start_date'):
            start_date = datetime.fromisoformat(request.args.get('start_date'))
        if request.args.get('end_date'):
            end_date = datetime.fromisoformat(request.args.get('end_date'))
        
        reports = analytics_db.get_daily_reports(
            camera_id=camera_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'count': len(reports),
            'reports': reports
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/reports/daily/generate', methods=['POST'])
def generate_daily_report():
    """
    Generate daily report for a camera
    
    Request Body:
        - camera_id: Camera identifier (required)
        - date: Date (ISO format, optional - defaults to today)
    
    Returns:
        JSON with generated report
    """
    try:
        data = request.get_json() or {}
        
        camera_id = data.get('camera_id')
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'camera_id is required'
            }), 400
        
        date = datetime.utcnow()
        if data.get('date'):
            date = datetime.fromisoformat(data['date'])
        
        report = analytics_db.generate_daily_report(camera_id, date)
        
        return jsonify({
            'success': True,
            'report': report
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/reports/daily/export', methods=['POST'])
def export_daily_reports():
    """
    Export daily reports to CSV
    
    Request Body:
        - camera_id: Camera identifier (required)
        - start_date: Start date (optional)
        - end_date: End date (optional)
        - filename: Output filename (optional)
    
    Returns:
        CSV file download
    """
    try:
        data = request.get_json() or {}
        
        camera_id = data.get('camera_id')
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'camera_id is required'
            }), 400
        
        start_date = None
        end_date = None
        if data.get('start_date'):
            start_date = datetime.fromisoformat(data['start_date'])
        if data.get('end_date'):
            end_date = datetime.fromisoformat(data['end_date'])
        
        filename = data.get('filename')
        
        filepath = exporter.export_daily_reports(
            camera_id=camera_id,
            start_date=start_date,
            end_date=end_date,
            filename=filename
        )
        
        if not filepath:
            return jsonify({
                'success': False,
                'error': 'No daily reports to export'
            }), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=os.path.basename(filepath),
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# PERSON JOURNEY ENDPOINTS
# ============================================================================

@analytics_bp.route('/journeys/<int:persistent_id>', methods=['GET'])
def get_person_journey(persistent_id):
    """
    Get journey information for a specific person
    
    Path Parameters:
        - persistent_id: Person's unique identifier
    
    Returns:
        JSON with journey information
    """
    try:
        journey = analytics_db.get_person_journey(persistent_id)
        
        if not journey:
            return jsonify({
                'success': False,
                'error': 'Journey not found'
            }), 404
        
        return jsonify({
            'success': True,
            'journey': journey
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/journeys/<int:persistent_id>/update', methods=['POST'])
def update_journey(persistent_id):
    """
    Update journey information for a person
    
    Path Parameters:
        - persistent_id: Person's unique identifier
    
    Returns:
        JSON with updated journey
    """
    try:
        journey = analytics_db.update_person_journey(persistent_id)
        
        if not journey:
            return jsonify({
                'success': False,
                'error': 'No detections found for person'
            }), 404
        
        return jsonify({
            'success': True,
            'journey': journey
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/journeys/<int:persistent_id>/close', methods=['POST'])
def close_journey(persistent_id):
    """
    Mark a person's journey as inactive
    
    Path Parameters:
        - persistent_id: Person's unique identifier
    
    Returns:
        JSON with success status
    """
    try:
        analytics_db.close_person_journey(persistent_id)
        
        return jsonify({
            'success': True,
            'message': f'Journey closed for person {persistent_id}'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/journeys/export', methods=['POST'])
def export_journeys():
    """
    Export person journeys to CSV
    
    Request Body:
        - start_time: Start time filter (optional)
        - end_time: End time filter (optional)
        - is_active: Active status filter (optional)
        - filename: Output filename (optional)
    
    Returns:
        CSV file download
    """
    try:
        data = request.get_json() or {}
        
        start_time = None
        end_time = None
        if data.get('start_time'):
            start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            end_time = datetime.fromisoformat(data['end_time'])
        
        is_active = data.get('is_active')
        filename = data.get('filename')
        
        filepath = exporter.export_person_journeys(
            start_time=start_time,
            end_time=end_time,
            is_active=is_active,
            filename=filename
        )
        
        if not filepath:
            return jsonify({
                'success': False,
                'error': 'No journeys to export'
            }), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=os.path.basename(filepath),
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# PASSENGER FLOW TRENDS - NEW ENDPOINTS
# ============================================================================

@analytics_bp.route('/trends/arrival-departure', methods=['GET'])
def get_arrival_departure_trends():
    """
    Get passenger arrival and departure trends
    
    Query Parameters:
        - camera_id: Camera identifier (optional, for specific zones)
        - fence_name: Geo-fence name (e.g., "Entrance", "Exit", "Platform")
        - start_time: Start timestamp (ISO format)
        - end_time: End timestamp (ISO format)
        - interval: Time interval ('hour', 'day', 'week') - default: 'hour'
    
    Returns:
        JSON with arrival/departure counts by time interval
    """
    try:
        camera_id = request.args.get('camera_id')
        fence_name = request.args.get('fence_name')
        interval = request.args.get('interval', 'hour')
        
        # Parse time range (default: last 24 hours)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        if request.args.get('start_time'):
            start_time = datetime.fromisoformat(request.args.get('start_time'))
        if request.args.get('end_time'):
            end_time = datetime.fromisoformat(request.args.get('end_time'))
        
        with analytics_db.session_scope() as session:
            # Build base query
            query = session.query(PersonJourney).filter(
                PersonJourney.first_seen >= start_time,
                PersonJourney.first_seen <= end_time
            )
            
            # Filter by camera if specified
            if camera_id:
                # Filter journeys that include this camera
                journeys_with_camera = []
                for journey in query.all():
                    if camera_id in (journey.cameras_visited or []):
                        journeys_with_camera.append(journey)
                journeys = journeys_with_camera
            else:
                journeys = query.all()
            
            # Aggregate by time interval
            trends = defaultdict(lambda: {
                'arrivals': 0,
                'departures': 0
            })
            
            # Define interval format
            if interval == 'hour':
                time_format = lambda dt: dt.strftime('%Y-%m-%d %H:00')
            elif interval == 'day':
                time_format = lambda dt: dt.strftime('%Y-%m-%d')
            elif interval == 'week':
                time_format = lambda dt: dt.strftime('%Y-W%W')
            else:
                time_format = lambda dt: dt.strftime('%Y-%m-%d %H:00')
            
            # Count arrivals and departures
            for journey in journeys:
                # Arrival (first_seen)
                arrival_interval = time_format(journey.first_seen)
                trends[arrival_interval]['arrivals'] += 1
                
                # Departure (last_seen if inactive)
                if not journey.is_active and journey.last_seen:
                    departure_interval = time_format(journey.last_seen)
                    trends[departure_interval]['departures'] += 1
            
            # Convert to list and calculate metrics
            trend_list = []
            arrival_counts = []
            departure_counts = []
            
            for interval_time in sorted(trends.keys()):
                data = trends[interval_time]
                net_flow = data['arrivals'] - data['departures']
                
                trend_list.append({
                    'interval': interval_time,
                    'arrivals': data['arrivals'],
                    'departures': data['departures'],
                    'net_flow': net_flow,
                    'peak_hour': False  # Will mark later
                })
                
                arrival_counts.append(data['arrivals'])
                departure_counts.append(data['departures'])
            
            # Identify peak hours
            if arrival_counts:
                peak_arrivals = max(arrival_counts)
                peak_departures = max(departure_counts)
                
                for trend in trend_list:
                    if trend['arrivals'] == peak_arrivals or trend['departures'] == peak_departures:
                        trend['peak_hour'] = True
            
            # Calculate summary statistics
            total_arrivals = sum(arrival_counts)
            total_departures = sum(departure_counts)
            
            peak_arrival_interval = max(trends.items(), key=lambda x: x[1]['arrivals'])[0] if trends else None
            peak_departure_interval = max(trends.items(), key=lambda x: x[1]['departures'])[0] if trends else None
            
            summary = {
                'total_arrivals': total_arrivals,
                'total_departures': total_departures,
                'net_change': total_arrivals - total_departures,
                'peak_arrival_interval': peak_arrival_interval,
                'peak_departure_interval': peak_departure_interval,
                'avg_arrivals': round(statistics.mean(arrival_counts), 1) if arrival_counts else 0,
                'avg_departures': round(statistics.mean(departure_counts), 1) if departure_counts else 0,
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'interval': interval
                }
            }
        
        return jsonify({
            'success': True,
            'trends': trend_list,
            'summary': summary
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/trends/peak-crowding', methods=['GET'])
def get_peak_crowding_times():
    """
    Identify peak crowding times and occupancy patterns
    
    Query Parameters:
        - camera_id: Camera identifier (required)
        - start_date: Start date (ISO format)
        - end_date: End date (ISO format)
        - threshold: Occupancy threshold for "crowded" (default: 30)
    
    Returns:
        JSON with peak crowding analysis
    """
    try:
        camera_id = request.args.get('camera_id')
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'camera_id is required'
            }), 400
        
        threshold = request.args.get('threshold', default=30, type=int)
        
        # Parse date range (default: last 7 days)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        if request.args.get('start_date'):
            start_date = datetime.fromisoformat(request.args.get('start_date'))
        if request.args.get('end_date'):
            end_date = datetime.fromisoformat(request.args.get('end_date'))
        
        with analytics_db.session_scope() as session:
            # Get hourly occupancy data
            hourly_data = defaultdict(lambda: {
                'counts': [],
                'max': 0,
                'days_crowded': 0
            })
            
            daily_patterns = defaultdict(lambda: {
                'hourly_peaks': [],
                'max_occupancy': 0
            })
            
            # Query detections grouped by hour
            current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            while current_date <= end_date:
                for hour in range(24):
                    hour_start = current_date.replace(hour=hour)
                    hour_end = hour_start + timedelta(hours=1)
                    
                    # Count unique persons in this hour
                    occupancy = session.query(
                        func.count(func.distinct(PersonDetection.persistent_id))
                    ).filter(
                        PersonDetection.camera_id == camera_id,
                        PersonDetection.timestamp >= hour_start,
                        PersonDetection.timestamp < hour_end
                    ).scalar() or 0
                    
                    if occupancy > 0:
                        hourly_data[hour]['counts'].append(occupancy)
                        hourly_data[hour]['max'] = max(hourly_data[hour]['max'], occupancy)
                        
                        if occupancy >= threshold:
                            hourly_data[hour]['days_crowded'] += 1
                        
                        # Track daily patterns
                        day_name = current_date.strftime('%A').lower()
                        daily_patterns[day_name]['hourly_peaks'].append(occupancy)
                        daily_patterns[day_name]['max_occupancy'] = max(
                            daily_patterns[day_name]['max_occupancy'], 
                            occupancy
                        )
                
                current_date += timedelta(days=1)
            
            # Calculate peak times
            peak_times = []
            for hour in range(24):
                if hourly_data[hour]['counts']:
                    avg_occupancy = statistics.mean(hourly_data[hour]['counts'])
                    
                    # Determine crowd level
                    if avg_occupancy >= threshold * 1.5:
                        crowd_level = 'critical'
                    elif avg_occupancy >= threshold:
                        crowd_level = 'high'
                    elif avg_occupancy >= threshold * 0.7:
                        crowd_level = 'medium'
                    else:
                        crowd_level = 'low'
                    
                    peak_times.append({
                        'hour': hour,
                        'time_label': f"{hour:02d}:00",
                        'avg_occupancy': round(avg_occupancy, 1),
                        'max_occupancy': hourly_data[hour]['max'],
                        'days_crowded': hourly_data[hour]['days_crowded'],
                        'crowd_level': crowd_level
                    })
            
            # Sort by average occupancy
            peak_times.sort(key=lambda x: x['avg_occupancy'], reverse=True)
            
            # Calculate daily patterns
            daily_summary = {}
            for day, data in daily_patterns.items():
                if data['hourly_peaks']:
                    peak_hour = data['hourly_peaks'].index(max(data['hourly_peaks']))
                    daily_summary[day] = {
                        'peak_hour': peak_hour,
                        'peak_hour_label': f"{peak_hour:02d}:00",
                        'avg_peak': round(statistics.mean(data['hourly_peaks']), 1),
                        'max_occupancy': data['max_occupancy']
                    }
            
            # Overall summary
            all_peaks = [t['avg_occupancy'] for t in peak_times]
            overall_peak_hour = peak_times[0] if peak_times else None
            
            # Find peak day
            peak_day = max(daily_summary.items(), 
                          key=lambda x: x[1]['avg_peak'])[0] if daily_summary else None
            
            summary = {
                'overall_peak_hour': overall_peak_hour['hour'] if overall_peak_hour else None,
                'overall_peak_hour_label': overall_peak_hour['time_label'] if overall_peak_hour else None,
                'overall_peak_day': peak_day.title() if peak_day else None,
                'avg_daily_peak': round(statistics.mean(all_peaks), 1) if all_peaks else 0,
                'max_recorded': max(all_peaks) if all_peaks else 0,
                'crowding_threshold': threshold,
                'hours_above_threshold': sum(1 for t in peak_times if t['avg_occupancy'] >= threshold)
            }
        
        return jsonify({
            'success': True,
            'peak_times': peak_times[:10],  # Top 10 peak hours
            'daily_patterns': daily_summary,
            'summary': summary,
            'analysis_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'days_analyzed': (end_date - start_date).days
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/trends/wait-times', methods=['GET'])
def get_average_wait_times():
    """
    Calculate average wait times before boarding/departure
    
    Query Parameters:
        - camera_id: Camera identifier (optional)
        - fence_name: Boarding zone name (e.g., "Platform", "Gate")
        - start_time: Start timestamp (ISO format)
        - end_time: End timestamp (ISO format)
    
    Returns:
        JSON with wait time analysis
    """
    try:
        camera_id = request.args.get('camera_id')
        fence_name = request.args.get('fence_name')
        
        # Parse time range (default: last 24 hours)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        if request.args.get('start_time'):
            start_time = datetime.fromisoformat(request.args.get('start_time'))
        if request.args.get('end_time'):
            end_time = datetime.fromisoformat(request.args.get('end_time'))
        
        with analytics_db.session_scope() as session:
            # Get journeys in time range
            query = session.query(PersonJourney).filter(
                PersonJourney.first_seen >= start_time,
                PersonJourney.last_seen <= end_time,
                PersonJourney.is_active == False  # Only completed journeys
            )
            
            # Filter by camera if specified
            if camera_id:
                journeys_with_camera = []
                for journey in query.all():
                    if camera_id in (journey.cameras_visited or []):
                        journeys_with_camera.append(journey)
                journeys = journeys_with_camera
            else:
                journeys = query.all()
            
            if not journeys:
                return jsonify({
                    'success': False,
                    'error': 'No completed journeys found in time range'
                }), 404
            
            # Calculate wait times (duration in zone)
            wait_times = []
            by_hour = defaultdict(list)
            by_zone = defaultdict(list)
            
            for journey in journeys:
                # Use total_duration_seconds as wait time
                wait_seconds = journey.total_duration_seconds
                
                if wait_seconds > 0:
                    wait_times.append(wait_seconds)
                    
                    # Group by hour
                    hour = journey.first_seen.hour
                    by_hour[hour].append(wait_seconds)
                    
                    # Group by zone (use first camera visited)
                    if journey.cameras_visited:
                        zone = journey.cameras_visited[0]
                        by_zone[zone].append(wait_seconds)
            
            if not wait_times:
                return jsonify({
                    'success': False,
                    'error': 'No valid wait times calculated'
                }), 404
            
            # Calculate statistics
            wait_times.sort()
            n = len(wait_times)
            
            percentiles = {
                'p25': wait_times[int(n * 0.25)],
                'p50': wait_times[int(n * 0.50)],
                'p75': wait_times[int(n * 0.75)],
                'p90': wait_times[int(n * 0.90)]
            }
            
            # Time of day breakdown
            hourly_breakdown = []
            for hour in sorted(by_hour.keys()):
                times = by_hour[hour]
                hourly_breakdown.append({
                    'hour': hour,
                    'time_label': f"{hour:02d}:00",
                    'avg_wait_seconds': round(statistics.mean(times), 1),
                    'avg_wait_minutes': round(statistics.mean(times) / 60, 1),
                    'sample_size': len(times)
                })
            
            # Zone breakdown
            zone_breakdown = {}
            for zone, times in by_zone.items():
                zone_breakdown[zone] = {
                    'avg_wait_seconds': round(statistics.mean(times), 1),
                    'avg_wait_minutes': round(statistics.mean(times) / 60, 1),
                    'count': len(times)
                }
            
            summary = {
                'avg_wait_seconds': round(statistics.mean(wait_times), 1),
                'avg_wait_minutes': round(statistics.mean(wait_times) / 60, 1),
                'median_wait_seconds': percentiles['p50'],
                'median_wait_minutes': round(percentiles['p50'] / 60, 1),
                'min_wait_seconds': min(wait_times),
                'max_wait_seconds': max(wait_times),
                'std_dev_seconds': round(statistics.stdev(wait_times), 1) if len(wait_times) > 1 else 0,
                'percentiles': percentiles,
                'total_passengers': n
            }
        
        return jsonify({
            'success': True,
            'wait_times': summary,
            'by_time_of_day': hourly_breakdown,
            'by_zone': zone_breakdown,
            'analysis_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# BATCH EXPORT ENDPOINTS
# ============================================================================

@analytics_bp.route('/export/all', methods=['POST'])
def export_all_data():
    """
    Export all analytics data for a time period
    
    Request Body:
        - camera_id: Camera identifier (required)
        - start_time: Start timestamp (required)
        - end_time: End timestamp (required)
        - prefix: Optional prefix for filenames
    
    Returns:
        JSON with paths to all exported files
    """
    try:
        data = request.get_json() or {}
        
        camera_id = data.get('camera_id')
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'camera_id is required'
            }), 400
        
        if not data.get('start_time') or not data.get('end_time'):
            return jsonify({
                'success': False,
                'error': 'start_time and end_time are required'
            }), 400
        
        start_time = datetime.fromisoformat(data['start_time'])
        end_time = datetime.fromisoformat(data['end_time'])
        prefix = data.get('prefix', '')
        
        exports = exporter.export_all_data(
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            prefix=prefix
        )
        
        return jsonify({
            'success': True,
            'exports': {
                key: os.path.basename(path) if path else None
                for key, path in exports.items()
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/export/summary', methods=['POST'])
def export_summary_report():
    """
    Export a comprehensive summary report
    
    Request Body:
        - camera_id: Camera identifier (required)
        - start_date: Start date (required)
        - end_date: End date (required)
        - filename: Output filename (optional)
    
    Returns:
        CSV file download
    """
    try:
        data = request.get_json() or {}
        
        camera_id = data.get('camera_id')
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'camera_id is required'
            }), 400
        
        if not data.get('start_date') or not data.get('end_date'):
            return jsonify({
                'success': False,
                'error': 'start_date and end_date are required'
            }), 400
        
        start_date = datetime.fromisoformat(data['start_date'])
        end_date = datetime.fromisoformat(data['end_date'])
        filename = data.get('filename')
        
        filepath = exporter.export_summary_report(
            camera_id=camera_id,
            start_date=start_date,
            end_date=end_date,
            filename=filename
        )
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=os.path.basename(filepath),
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================


@analytics_bp.route('/dashboard/analytics', methods=['GET'])
def get_dashboard_analytics():
    """
    Get analytics data for AnalyticsDashboard component
    Returns: incident trends, hotspots, key metrics
    """
    try:
        with analytics_db.session_scope() as session:
            # Get total incidents (behaviors)
            total_incidents = session.query(BehaviorEvent).count()
            
            # Get tracked persons (unique persistent IDs)
            tracked_persons = session.query(func.count(func.distinct(PersonDetection.persistent_id))).scalar()
            
            # Get unique visitors (closed journeys)
            unique_visitors = session.query(PersonJourney).filter(
                PersonJourney.is_active == False
            ).count()
            
            # Get incident hotspots (top 5 zones by behavior count)
            hotspot_results = session.query(
                PersonDetection.fence_name,
                func.count(BehaviorEvent.id).label('count')
            ).join(BehaviorEvent).filter(
                PersonDetection.fence_name.isnot(None)
            ).group_by(PersonDetection.fence_name).order_by(
                func.count(BehaviorEvent.id).desc()
            ).limit(5).all()
            
            hotspots = [
                {'zone': row[0], 'count': row[1]} 
                for row in hotspot_results
            ]
            
            # Get incident trends (last 12 hours)
            twelve_hours_ago = datetime.utcnow() - timedelta(hours=12)
            trend_results = session.query(
                func.strftime('%H:00', BehaviorEvent.timestamp).label('time'),
                func.count(BehaviorEvent.id).label('incidents')
            ).filter(
                BehaviorEvent.timestamp > twelve_hours_ago
            ).group_by('time').order_by('time').all()
            
            trends = [
                {'time': row[0], 'incidents': row[1]} 
                for row in trend_results
            ]
        
        return jsonify({
            'success': True,
            'analytics': {
                'totalIncidents': total_incidents,
                'avgResponseTime': 3.2,
                'trackedPersons': tracked_persons,
                'uniqueVisitors': unique_visitors,
                'incidentTrends': trends,
                'incidentHotspots': hotspots
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/dashboard/anomalies', methods=['GET'])
def get_dashboard_anomalies():
    """
    Get recent anomaly logs for AnomalyLogsPanel
    Returns: Recent behavior events with ALL detection details including metadata
    ⭐ FIXED: Now properly extracts ALL fields from meta_data JSON column
    """
    try:
        with analytics_db.session_scope() as session:
            # Get recent behaviors with detection info
            behaviors = session.query(BehaviorEvent).join(PersonDetection).order_by(
                BehaviorEvent.timestamp.desc()
            ).limit(50).all()
            
            anomalies = []
            for b in behaviors:
                # Start with base data from database columns
                anomaly = {
                    'id': b.id,
                    'type': b.behavior_type,
                    'severity': b.severity,
                    'roi': b.detection.fence_name or 'Unknown Zone',
                    'timestamp': b.timestamp.isoformat() if b.timestamp else None,
                    'confidence': b.confidence,
                    'description': b.description,
                    'person_id': b.detection.persistent_id if b.detection else None
                }
                
                # Add position if available
                if b.position_x is not None and b.position_y is not None:
                    anomaly['position'] = [b.position_x, b.position_y]
                
                # ⭐ CRITICAL: Merge ALL metadata fields from JSON column
                # This includes: duration, velocity, confirmation_ratio, 
                # violent_person_ids, max_arm_velocity, movement_range, etc.
                if b.meta_data:
                    anomaly.update(b.meta_data)
                
                anomalies.append(anomaly)
        
        return jsonify({
            'success': True,
            'anomalies': anomalies
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/dashboard/roi', methods=['GET'])
def get_dashboard_roi():
    """
    Get ROI occupancy data by geo-fence
    Returns: Current occupancy per region of interest
    """
    try:
        with analytics_db.session_scope() as session:
            # Get recent detections (last 5 minutes) grouped by fence
            five_min_ago = datetime.utcnow() - timedelta(minutes=5)
            
            roi_results = session.query(
                PersonDetection.fence_name,
                func.count(func.distinct(PersonDetection.persistent_id)).label('count')
            ).filter(
                PersonDetection.timestamp > five_min_ago,
                PersonDetection.in_geo_fence == True,
                PersonDetection.fence_name.isnot(None)
            ).group_by(PersonDetection.fence_name).all()
            
            roi_data = {}
            for row in roi_results:
                fence_name = row[0]
                count = row[1]
                
                # Calculate average dwell time (simplified)
                avg_dwell = session.query(
                    func.avg(PersonJourney.total_duration_seconds)
                ).join(PersonDetection).filter(
                    PersonDetection.fence_name == fence_name
                ).scalar() or 0
                
                roi_data[fence_name] = {
                    'count': count,
                    'capacity': 45,  # TODO: Get from geo-fence config
                    'avgDwellTime': round(avg_dwell / 60, 1)  # Convert to minutes
                }
        
        return jsonify({
            'success': True,
            'roiData': roi_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@analytics_bp.route('/dashboard/system-health', methods=['GET'])
def get_dashboard_system_health():
    """
    Get real-time system health metrics
    Returns: Real FPS, detection accuracy, uptime, active UIDs
    """
    try:
        with analytics_db.session_scope() as session:
            # Get metrics from last 60 seconds
            time_threshold = datetime.utcnow() - timedelta(seconds=60)
            
            # 1. Calculate actual FPS from recent detections
            recent_detections = session.query(PersonDetection).filter(
                PersonDetection.timestamp >= time_threshold
            ).all()
            
            if recent_detections:
                time_span = (datetime.utcnow() - min(d.timestamp for d in recent_detections)).total_seconds()
                unique_frames = len(set(d.frame_id for d in recent_detections))
                fps = round(unique_frames / time_span if time_span > 0 else 0, 1)
            else:
                fps = 0
            
            # 2. Get active unique tracked persons (persistent_id) in last 30 seconds
            active_threshold = datetime.utcnow() - timedelta(seconds=30)
            active_uids = session.query(
                func.count(func.distinct(PersonDetection.persistent_id))
            ).filter(
                PersonDetection.timestamp >= active_threshold
            ).scalar() or 0
            
            # 3. Calculate detection accuracy (detections with high confidence)
            high_confidence_count = session.query(func.count(PersonDetection.id)).filter(
                PersonDetection.timestamp >= time_threshold,
                PersonDetection.confidence >= 0.7
            ).scalar() or 0
            
            total_detections = len(recent_detections)
            detection_accuracy = round((high_confidence_count / total_detections * 100) if total_detections > 0 else 0, 1)
            
            # 4. Calculate ID switch rate (how often persistent_id changes per track_id)
            id_switches = 0
            if recent_detections:
                track_groups = {}
                for d in recent_detections:
                    if d.track_id not in track_groups:
                        track_groups[d.track_id] = set()
                    track_groups[d.track_id].add(d.persistent_id)
                
                # Count tracks with multiple persistent IDs
                id_switches = sum(1 for pids in track_groups.values() if len(pids) > 1)
                id_switch_rate = round((id_switches / len(track_groups) * 100) if track_groups else 0, 1)
            else:
                id_switch_rate = 0.0
            
            # 5. Calculate uptime (percentage of time with active detections in last 5 minutes)
            uptime_windows = 10  # 5 minutes / 30 seconds
            active_windows = 0
            
            for i in range(uptime_windows):
                window_start = datetime.utcnow() - timedelta(seconds=30 * (i + 1))
                window_end = datetime.utcnow() - timedelta(seconds=30 * i)
                
                window_detections = session.query(func.count(PersonDetection.id)).filter(
                    PersonDetection.timestamp >= window_start,
                    PersonDetection.timestamp < window_end
                ).scalar() or 0
                
                if window_detections > 0:
                    active_windows += 1
            
            uptime = round((active_windows / uptime_windows * 100), 1)
            
            # 6. Anomaly precision (percentage of high-severity behaviors)
            recent_behaviors = session.query(BehaviorEvent).filter(
                BehaviorEvent.timestamp >= time_threshold
            ).all()
            
            high_severity_count = sum(1 for b in recent_behaviors if b.severity in ['high', 'critical'])
            anomaly_precision = round((high_severity_count / len(recent_behaviors) * 100) if recent_behaviors else 0, 1)
        
        health = {
            'fps': fps,
            'detectionAccuracy': detection_accuracy,
            'anomalyPrecision': anomaly_precision,
            'uptime': uptime,
            'idSwitchRate': id_switch_rate,
            'activeUIDs': active_uids
        }
        
        return jsonify({
            'success': True,
            'health': health
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@analytics_bp.route('/dashboard/incidents', methods=['GET'])
def get_dashboard_incidents():
    """
    Get incident reports (high-severity behaviors)
    Returns: Active incidents/alerts for IncidentPanel
    """
    try:
        with analytics_db.session_scope() as session:
            # Get recent high-severity behaviors
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            
            behaviors = session.query(BehaviorEvent).join(PersonDetection).filter(
                BehaviorEvent.severity.in_(['high', 'critical']),
                BehaviorEvent.timestamp > one_hour_ago
            ).order_by(BehaviorEvent.timestamp.desc()).limit(50).all()
            
            incidents = []
            for b in behaviors:
                incidents.append({
                    'id': b.id,
                    'category': b.behavior_type.replace('_', ' ').title(),
                    'type': b.behavior_type,
                    'severity': b.severity,
                    'zone': b.detection.fence_name or 'Unknown',
                    'camera_id': b.detection.camera_id,
                    'timestamp': b.timestamp.isoformat() if b.timestamp else None,
                    'status': 'reported',
                    'tracked_ids': []
                })
        
        return jsonify({
            'success': True,
            'incidents': incidents
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@analytics_bp.route('/cleanup', methods=['POST'])
def cleanup_old_data():
    """
    Remove old analytics data
    
    Request Body:
        - days_to_keep: Number of days to keep (default: 30)
    
    Returns:
        JSON with cleanup statistics
    """
    try:
        data = request.get_json() or {}
        days_to_keep = data.get('days_to_keep', 30)
        
        results = analytics_db.cleanup_old_data(days_to_keep)
        
        return jsonify({
            'success': True,
            'days_kept': days_to_keep,
            'deleted': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/health', methods=['GET'])
def health_check():
    """
    Check API health status
    
    Returns:
        JSON with health status
    """
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    }), 200
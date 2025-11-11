# app/api/video/routes.py
from flask import Blueprint, jsonify, request
from flask_socketio import emit, join_room
from app.extensions import socketio
import base64
import time
import sqlite3
import json
from datetime import datetime
from .stream_manager import stream_manager

video_bp = Blueprint('video', __name__)

# REMOVE the try-except block entirely - we'll start cameras later
# Just keep the imports and route definitions

@video_bp.route('/cameras')
def get_cameras():
    return jsonify([
        {'id': cam_id,
         'active': cam['active'],
         'count': len(stream_manager.active_ids[cam_id])}
        for cam_id, cam in stream_manager.cameras.items()
    ])

@video_bp.route('/start/<camera_id>')
def start_camera(camera_id):
    success = stream_manager.start_stream(camera_id)
    return jsonify({'success': success})

# ============================================================================
# MULTI GEO-FENCE ENDPOINTS (NEW)
# ============================================================================

@video_bp.route('/geo-fences/<camera_id>', methods=['GET'])
def get_geo_fences(camera_id):
    """Get all geo-fences for a specific camera"""
    try:
        print(f"[API] GET geo-fences for camera: {camera_id}")
        fences = stream_manager.get_geo_fences(camera_id)
        print(f"[API] Returning {len(fences)} fences")
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'fences': fences
        })
    except Exception as e:
        print(f"[ERROR] get_geo_fences: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@video_bp.route('/geo-fences/<camera_id>', methods=['POST'])
def create_geo_fence(camera_id):
    """Create a new geo-fence for a camera"""
    try:
        print(f"[API] POST create geo-fence for camera: {camera_id}")
        data = request.get_json()
        print(f"[API] Received data: {data}")
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        name = data.get('name', 'Zone')
        points = data.get('points', [])
        
        print(f"[API] Name: {name}, Points count: {len(points)}")
        
        # Validate points
        if len(points) < 3:
            return jsonify({
                'success': False,
                'error': 'Geo-fence must have at least 3 points'
            }), 400
        
        # Validate point format
        for i, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return jsonify({
                    'success': False,
                    'error': f'Point {i+1} has invalid format. Expected [x, y], got {point}'
                }), 400
            
            try:
                x, y = float(point[0]), float(point[1])
                if x < 0 or y < 0:
                    return jsonify({
                        'success': False,
                        'error': f'Point {i+1} has negative coordinates'
                    }), 400
            except (ValueError, TypeError) as e:
                return jsonify({
                    'success': False,
                    'error': f'Point {i+1} has non-numeric coordinates: {e}'
                }), 400
        
        print(f"[API] Points validated, calling add_geo_fence...")
        fence_id = stream_manager.add_geo_fence(camera_id, name, points)
        print(f"[API] add_geo_fence returned: {fence_id}")
        
        # NEW: Also save to alerts database
        if fence_id:
            try:
                conn = sqlite3.connect('analytics.db')
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO geofences (id, name, camera_id, polygon_points, created_at, enabled)
                    VALUES (?, ?, ?, ?, ?, 1)
                """, (fence_id, name, camera_id, json.dumps(points), datetime.now().isoformat()))
                conn.commit()
                conn.close()
                print(f"[API] Saved geo-fence '{name}' (ID: {fence_id}) to alerts database")
            except Exception as db_error:
                print(f"[WARNING] Failed to save geo-fence to alerts database: {db_error}")
        
        if fence_id:
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'fence_id': fence_id,
                'message': f'Geo-fence "{name}" created successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create geo-fence - add_geo_fence returned None'
            }), 500
            
    except Exception as e:
        print(f"[ERROR] create_geo_fence: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@video_bp.route('/geo-fences/<camera_id>/<fence_id>', methods=['PUT'])
def update_geo_fence(camera_id, fence_id):
    """Update an existing geo-fence"""
    try:
        print(f"[API] PUT update geo-fence {fence_id} for camera: {camera_id}")
        data = request.get_json()
        print(f"[API] Update data: {data}")
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        points = data.get('points')
        name = data.get('name')
        enabled = data.get('enabled')
        
        # Validate points if provided
        if points is not None:
            if len(points) < 3:
                return jsonify({
                    'success': False,
                    'error': 'Geo-fence must have at least 3 points'
                }), 400
            
            for i, point in enumerate(points):
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    return jsonify({
                        'success': False,
                        'error': f'Point {i+1} has invalid format. Expected [x, y]'
                    }), 400
        
        print(f"[API] Calling update_geo_fence with points={points is not None}, name={name}, enabled={enabled}")
        success = stream_manager.update_geo_fence(camera_id, fence_id, points, name, enabled)
        print(f"[API] update_geo_fence returned: {success}")
        
        # NEW: Also update in alerts database
        if success:
            try:
                conn = sqlite3.connect('analytics.db')
                cursor = conn.cursor()
                
                update_parts = []
                update_values = []
                
                if points is not None:
                    update_parts.append("polygon_points = ?")
                    update_values.append(json.dumps(points))
                
                if name is not None:
                    update_parts.append("name = ?")
                    update_values.append(name)
                
                if enabled is not None:
                    update_parts.append("enabled = ?")
                    update_values.append(1 if enabled else 0)
                
                if update_parts:
                    update_parts.append("updated_at = ?")
                    update_values.append(datetime.now().isoformat())
                    
                    update_values.append(fence_id)
                    
                    query = f"UPDATE geofences SET {', '.join(update_parts)} WHERE id = ?"
                    cursor.execute(query, update_values)
                    conn.commit()
                    conn.close()
                    print(f"[API] Updated geo-fence in alerts database")
            except Exception as db_error:
                print(f"[WARNING] Failed to update geo-fence in alerts database: {db_error}")
        
        if success:
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'fence_id': fence_id,
                'message': 'Geo-fence updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Geo-fence not found or update failed'
            }), 404
            
    except Exception as e:
        print(f"[ERROR] update_geo_fence: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@video_bp.route('/geo-fences/<camera_id>/<fence_id>', methods=['DELETE'])
def delete_geo_fence(camera_id, fence_id):
    """Delete a geo-fence"""
    try:
        print(f"[API] DELETE geo-fence {fence_id} for camera: {camera_id}")
        success = stream_manager.remove_geo_fence(camera_id, fence_id)
        print(f"[API] remove_geo_fence returned: {success}")
        
        # NEW: Also delete from alerts database
        if success:
            try:
                conn = sqlite3.connect('analytics.db')
                cursor = conn.cursor()
                cursor.execute("DELETE FROM geofences WHERE id = ?", (fence_id,))
                conn.commit()
                conn.close()
                print(f"[API] Deleted geo-fence from alerts database")
            except Exception as db_error:
                print(f"[WARNING] Failed to delete geo-fence from alerts database: {db_error}")
        
        if success:
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'fence_id': fence_id,
                'message': 'Geo-fence deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Geo-fence not found'
            }), 404
            
    except Exception as e:
        print(f"[ERROR] delete_geo_fence: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@video_bp.route('/geo-fences/<camera_id>/<fence_id>/toggle', methods=['POST'])
def toggle_geo_fence_multi(camera_id, fence_id):
    """Toggle geo-fence enabled/disabled"""
    try:
        print(f"[API] POST toggle geo-fence {fence_id} for camera: {camera_id}")
        enabled = stream_manager.toggle_geo_fence(camera_id, fence_id)
        print(f"[API] toggle_geo_fence returned: {enabled}")
        
        # NEW: Also toggle in alerts database
        if enabled is not None:
            try:
                conn = sqlite3.connect('analytics.db')
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE geofences SET enabled = ?, updated_at = ? WHERE id = ?",
                    (1 if enabled else 0, datetime.now().isoformat(), fence_id)
                )
                conn.commit()
                conn.close()
                print(f"[API] Toggled geo-fence in alerts database")
            except Exception as db_error:
                print(f"[WARNING] Failed to toggle geo-fence in alerts database: {db_error}")
        
        if enabled is not None:
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'fence_id': fence_id,
                'enabled': enabled,
                'message': f'Geo-fence {"enabled" if enabled else "disabled"}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Geo-fence not found'
            }), 404
            
    except Exception as e:
        print(f"[ERROR] toggle_geo_fence_multi: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# LEGACY GEO-FENCE ENDPOINTS (KEPT FOR BACKWARD COMPATIBILITY)
# These can be removed once you migrate to multi geo-fence system
# ============================================================================

@video_bp.route('/geo-fence/<camera_id>', methods=['POST'])
def set_geo_fence_legacy(camera_id):
    """Legacy endpoint - creates/updates a single geo-fence"""
    try:
        print(f"[API] Legacy POST geo-fence for camera: {camera_id}")
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        points = data.get('points', [])
        
        # Support 4-point 2D mode only in legacy endpoint
        if len(points) != 4:
            return jsonify({
                'success': False, 
                'error': f'Must provide exactly 4 points, got {len(points)}'
            }), 400
        
        # Validate point format
        for i, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return jsonify({
                    'success': False,
                    'error': f'Point {i+1} has invalid format. Expected [x, y]'
                }), 400
            
            try:
                x, y = float(point[0]), float(point[1])
                if x < 0 or y < 0:
                    return jsonify({
                        'success': False,
                        'error': f'Point {i+1} has negative coordinates'
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': f'Point {i+1} has non-numeric coordinates'
                }), 400
        
        # Check if camera has any fences, if not create first one, otherwise update first one
        existing_fences = stream_manager.get_geo_fences(camera_id)
        
        if len(existing_fences) == 0:
            # Create new fence
            fence_id = stream_manager.add_geo_fence(camera_id, "Main Zone", points)
            success = fence_id is not None
            
            # Save to alerts database
            if fence_id:
                try:
                    conn = sqlite3.connect('analytics.db')
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO geofences (id, name, camera_id, polygon_points, created_at, enabled)
                        VALUES (?, ?, ?, ?, ?, 1)
                    """, (fence_id, "Main Zone", camera_id, json.dumps(points), datetime.now().isoformat()))
                    conn.commit()
                    conn.close()
                except Exception as db_error:
                    print(f"[WARNING] Failed to save geo-fence to alerts database: {db_error}")
        else:
            # Update first fence
            fence_id = existing_fences[0]['id']
            success = stream_manager.update_geo_fence(camera_id, fence_id, points=points)
            
            # Update in alerts database
            if success:
                try:
                    conn = sqlite3.connect('analytics.db')
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE geofences SET polygon_points = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(points), datetime.now().isoformat(), fence_id)
                    )
                    conn.commit()
                    conn.close()
                except Exception as db_error:
                    print(f"[WARNING] Failed to update geo-fence in alerts database: {db_error}")
        
        if success:
            return jsonify({
                'success': True,
                'points': points,
                'message': 'Geo-fence updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Camera not found or invalid camera ID'
            }), 404
            
    except Exception as e:
        print(f"[ERROR] set_geo_fence_legacy: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@video_bp.route('/geo-fence/<camera_id>/toggle', methods=['POST'])
def toggle_geo_fence_legacy(camera_id):
    """Legacy endpoint - toggles first geo-fence"""
    try:
        print(f"[API] Legacy toggle geo-fence for camera: {camera_id}")
        data = request.json
        if data is None:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        enabled = data.get('enabled', True)
        
        if not isinstance(enabled, bool):
            return jsonify({
                'success': False,
                'error': 'enabled must be boolean (true/false)'
            }), 400
        
        # Get first fence and toggle it
        existing_fences = stream_manager.get_geo_fences(camera_id)
        
        if len(existing_fences) == 0:
            return jsonify({
                'success': False,
                'error': 'No geo-fence found for this camera'
            }), 404
        
        fence_id = existing_fences[0]['id']
        current_state = existing_fences[0]['enabled']
        
        # Only toggle if needed
        if current_state != enabled:
            result = stream_manager.toggle_geo_fence(camera_id, fence_id)
            success = result is not None
            
            # Update in alerts database
            if success:
                try:
                    conn = sqlite3.connect('analytics.db')
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE geofences SET enabled = ?, updated_at = ? WHERE id = ?",
                        (1 if enabled else 0, datetime.now().isoformat(), fence_id)
                    )
                    conn.commit()
                    conn.close()
                except Exception as db_error:
                    print(f"[WARNING] Failed to update geo-fence in alerts database: {db_error}")
        else:
            success = True
        
        if success:
            return jsonify({
                'success': True,
                'enabled': enabled,
                'message': f'Geo-fence {"enabled" if enabled else "disabled"}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to toggle geo-fence'
            }), 500
            
    except Exception as e:
        print(f"[ERROR] toggle_geo_fence_legacy: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@video_bp.route('/geo-fence/<camera_id>', methods=['GET'])
def get_geo_fence_legacy(camera_id):
    """Legacy endpoint - returns first geo-fence"""
    try:
        print(f"[API] Legacy GET geo-fence for camera: {camera_id}")
        fences = stream_manager.get_geo_fences(camera_id)
        
        if len(fences) > 0:
            # Return first fence in legacy format
            first_fence = fences[0]
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'points': first_fence['points'],
                'enabled': first_fence['enabled']
            })
        else:
            # Return empty config if no fences exist
            return jsonify({
                'success': True,
                'camera_id': camera_id,
                'points': [],
                'enabled': True
            })
    except Exception as e:
        print(f"[ERROR] get_geo_fence_legacy: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@video_bp.route('/geo-fence/<camera_id>/reset', methods=['POST'])
def reset_geo_fence(camera_id):
    """Reset to default geo-fence"""
    try:
        print(f"[API] Reset geo-fence for camera: {camera_id}")
        if camera_id not in stream_manager.cameras:
            return jsonify({
                'success': False,
                'error': 'Camera not found'
            }), 404
        
        width, height = 960, 540
        margin_x = width // 6
        margin_y = height // 6
        
        default_points = [
            [margin_x, margin_y],
            [width - margin_x, margin_y],
            [width - margin_x, height - margin_y],
            [margin_x, height - margin_y]
        ]
        
        # Clear all existing fences and create default one
        existing_fences = stream_manager.get_geo_fences(camera_id)
        for fence in existing_fences:
            stream_manager.remove_geo_fence(camera_id, fence['id'])
            
            # Delete from alerts database
            try:
                conn = sqlite3.connect('analytics.db')
                cursor = conn.cursor()
                cursor.execute("DELETE FROM geofences WHERE id = ?", (fence['id'],))
                conn.commit()
                conn.close()
            except Exception as db_error:
                print(f"[WARNING] Failed to delete geo-fence from alerts database: {db_error}")
        
        # Create new default fence
        fence_id = stream_manager.add_geo_fence(camera_id, "Main Zone", default_points)
        
        # Save to alerts database
        if fence_id:
            try:
                conn = sqlite3.connect('analytics.db')  # ✅ FIXED: was "cconn"
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO geofences (id, name, camera_id, polygon_points, created_at, enabled)
                    VALUES (?, ?, ?, ?, ?, 1)
                """, (fence_id, "Main Zone", camera_id, json.dumps(default_points), datetime.now().isoformat()))
                conn.commit()
                conn.close()
            except Exception as db_error:
                print(f"[WARNING] Failed to save geo-fence to alerts database: {db_error}")
        
        if fence_id:
            return jsonify({
                'success': True,
                'points': default_points,
                'message': 'Geo-fence reset to default'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to reset geo-fence'
            }), 500
            
    except Exception as e:
        print(f"[ERROR] reset_geo_fence: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

# ============================================================================
# SOCKET.IO EVENTS
# ============================================================================

@socketio.on('join_camera')
def on_join_camera(data):
    camera_id = data.get('camera_id')
    if camera_id:
        join_room(f'camera_{camera_id}')
        emit('joined_camera', {'camera_id': camera_id})
        print(f"[Socket] Client joined camera room: {camera_id}")

@socketio.on('update_geo_fences')
def on_update_geo_fences(data):
    """Socket event to notify clients of geo-fence updates"""
    try:
        camera_id = data.get('camera_id')
        print(f"[Socket] update_geo_fences for camera: {camera_id}")
        
        if not camera_id:
            emit('error', {'message': 'camera_id is required'})
            return
        
        # Broadcast to all clients in the room
        socketio.emit('geo_fences_updated', {
            'camera_id': camera_id,
            'timestamp': time.time()
        }, room=f'camera_{camera_id}')
        
        emit('success', {'message': 'Geo-fences updated successfully'})
        print(f"[Socket] Broadcasted geo_fences_updated for camera: {camera_id}")
            
    except Exception as e:
        print(f"[ERROR] on_update_geo_fences: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': f'Server error: {str(e)}'})

def initialize_cameras():
    """Initialize cameras after socketio is ready"""
    try:
        print("[Video Routes] Initializing cameras...")
        stream_manager.add_camera("CAM001", 0)  # Match frontend
        stream_manager.add_camera("CAM002", 1)  # Match frontend
        
        # Start first camera
        success = stream_manager.start_stream("CAM001")
        if success:
            print("[Video Routes] ✅ CAM001 started successfully")
        else:
            print("[Video Routes] ❌ Failed to start CAM001")
            
        print(f"[Video Routes] Cameras configured:")
        print(f"  - CAM001: source={stream_manager.cameras.get('CAM001', {}).get('source')}, active={stream_manager.cameras.get('CAM001', {}).get('active')}")
        print(f"  - CAM002: source={stream_manager.cameras.get('CAM002', {}).get('source')}, active={stream_manager.cameras.get('CAM002', {}).get('active')}")
    except Exception as e:
        print(f"[Video Routes] ❌ Camera initialization error: {e}")
        import traceback
        traceback.print_exc()
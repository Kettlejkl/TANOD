# app/api/auth/routes.py
"""
Flask routes for user authentication
Handles user registration, login, and session management
"""

from flask import Blueprint, jsonify, request, session
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from functools import wraps
import secrets

from .database import auth_db
from .models import User, UserSession

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# ============================================================================
# AUTHENTICATION DECORATOR
# ============================================================================

def require_auth(f):
    """Decorator to require authentication for endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({
                'success': False,
                'error': 'Authentication required'
            }), 401
        
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
        
        user = auth_db.verify_session_token(token)
        
        if not user:
            return jsonify({
                'success': False,
                'error': 'Invalid or expired token'
            }), 401
        
        # Add user to request context
        request.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function


# ============================================================================
# REGISTRATION & LOGIN
# ============================================================================

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Register a new user
    
    Request Body:
        - username: Username (required, unique)
        - email: Email address (required, unique)
        - password: Password (required, min 6 chars)
        - role: User role (optional, default: 'operator')
    
    Returns:
        JSON with user data and session token
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('username'):
            return jsonify({
                'success': False,
                'error': 'Username is required'
            }), 400
        
        if not data.get('email'):
            return jsonify({
                'success': False,
                'error': 'Email is required'
            }), 400
        
        if not data.get('password'):
            return jsonify({
                'success': False,
                'error': 'Password is required'
            }), 400
        
        # Validate password length
        if len(data['password']) < 6:
            return jsonify({
                'success': False,
                'error': 'Password must be at least 6 characters'
            }), 400
        
        # Validate email format
        import re
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_pattern, data['email']):
            return jsonify({
                'success': False,
                'error': 'Invalid email format'
            }), 400
        
        # Check if username exists
        if auth_db.get_user_by_username(data['username']):
            return jsonify({
                'success': False,
                'error': 'Username already exists'
            }), 409
        
        # Check if email exists
        if auth_db.get_user_by_email(data['email']):
            return jsonify({
                'success': False,
                'error': 'Email already registered'
            }), 409
        
        # Hash password
        password_hash = generate_password_hash(data['password'])
        
        # Create user
        user = auth_db.create_user(
            username=data['username'],
            email=data['email'],
            password_hash=password_hash,
            role=data.get('role', 'operator')
        )
        
        # Create session token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=7)
        
        auth_db.create_session(
            user_id=user['id'],
            token=token,
            expires_at=expires_at
        )
        
        return jsonify({
            'success': True,
            'message': 'Registration successful',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            },
            'token': token,
            'expires_at': expires_at.isoformat()
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Login user
    
    Request Body:
        - username: Username (required)
        - password: Password (required)
    
    Returns:
        JSON with user data and session token
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('username'):
            return jsonify({
                'success': False,
                'error': 'Username is required'
            }), 400
        
        if not data.get('password'):
            return jsonify({
                'success': False,
                'error': 'Password is required'
            }), 400
        
        # Get user
        user = auth_db.get_user_by_username(data['username'])
        
        if not user:
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        # Check password
        if not check_password_hash(user['password_hash'], data['password']):
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        # Update last login
        auth_db.update_last_login(user['id'])
        
        # Create session token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=7)
        
        auth_db.create_session(
            user_id=user['id'],
            token=token,
            expires_at=expires_at
        )
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            },
            'token': token,
            'expires_at': expires_at.isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@auth_bp.route('/logout', methods=['POST'])
@require_auth
def logout():
    """
    Logout user (invalidate token)
    
    Headers:
        - Authorization: Bearer {token}
    
    Returns:
        JSON with success status
    """
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        auth_db.delete_session(token)
        
        return jsonify({
            'success': True,
            'message': 'Logout successful'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# USER PROFILE
# ============================================================================

@auth_bp.route('/me', methods=['GET'])
@require_auth
def get_current_user():
    """
    Get current user profile
    
    Headers:
        - Authorization: Bearer {token}
    
    Returns:
        JSON with user data
    """
    try:
        user = request.current_user
        
        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role'],
                'created_at': user['created_at'],
                'last_login': user['last_login']
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@auth_bp.route('/verify', methods=['POST'])
def verify_token():
    """
    Verify if a token is valid
    
    Request Body:
        - token: Session token
    
    Returns:
        JSON with validity status and user data
    """
    try:
        data = request.get_json()
        token = data.get('token')
        
        if not token:
            return jsonify({
                'success': False,
                'valid': False,
                'error': 'Token is required'
            }), 400
        
        user = auth_db.verify_session_token(token)
        
        if not user:
            return jsonify({
                'success': True,
                'valid': False
            }), 200
        
        return jsonify({
            'success': True,
            'valid': True,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# PASSWORD MANAGEMENT
# ============================================================================

@auth_bp.route('/change-password', methods=['POST'])
@require_auth
def change_password():
    """
    Change user password
    
    Headers:
        - Authorization: Bearer {token}
    
    Request Body:
        - current_password: Current password (required)
        - new_password: New password (required, min 6 chars)
    
    Returns:
        JSON with success status
    """
    try:
        data = request.get_json()
        user = request.current_user
        
        if not data.get('current_password'):
            return jsonify({
                'success': False,
                'error': 'Current password is required'
            }), 400
        
        if not data.get('new_password'):
            return jsonify({
                'success': False,
                'error': 'New password is required'
            }), 400
        
        if len(data['new_password']) < 6:
            return jsonify({
                'success': False,
                'error': 'New password must be at least 6 characters'
            }), 400
        
        # Verify current password
        if not check_password_hash(user['password_hash'], data['current_password']):
            return jsonify({
                'success': False,
                'error': 'Current password is incorrect'
            }), 401
        
        # Update password
        new_hash = generate_password_hash(data['new_password'])
        auth_db.update_password(user['id'], new_hash)
        
        return jsonify({
            'success': True,
            'message': 'Password changed successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@auth_bp.route('/users', methods=['GET'])
@require_auth
def list_users():
    """
    List all users (admin only)
    
    Headers:
        - Authorization: Bearer {token}
    
    Returns:
        JSON with list of users
    """
    try:
        user = request.current_user
        
        # Check if admin
        if user['role'] != 'admin':
            return jsonify({
                'success': False,
                'error': 'Admin access required'
            }), 403
        
        users = auth_db.get_all_users()
        
        return jsonify({
            'success': True,
            'count': len(users),
            'users': users
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@auth_bp.route('/users/<int:user_id>', methods=['DELETE'])
@require_auth
def delete_user(user_id):
    """
    Delete a user (admin only)
    
    Headers:
        - Authorization: Bearer {token}
    
    Path Parameters:
        - user_id: User ID to delete
    
    Returns:
        JSON with success status
    """
    try:
        user = request.current_user
        
        # Check if admin
        if user['role'] != 'admin':
            return jsonify({
                'success': False,
                'error': 'Admin access required'
            }), 403
        
        # Prevent self-deletion
        if user['id'] == user_id:
            return jsonify({
                'success': False,
                'error': 'Cannot delete your own account'
            }), 400
        
        auth_db.delete_user(user_id)
        
        return jsonify({
            'success': True,
            'message': f'User {user_id} deleted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# HEALTH CHECK
# ============================================================================

@auth_bp.route('/health', methods=['GET'])
def health_check():
    """Check auth API health"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    }), 200
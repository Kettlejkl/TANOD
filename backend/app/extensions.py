# app/extensions.py
"""
Flask extensions initialization
Prevents circular imports by creating extensions here first
"""

from flask_socketio import SocketIO

# Initialize SocketIO
socketio = SocketIO(
    cors_allowed_origins="*",  # Allow all origins for development
    async_mode='threading',     # Use threading mode
    logger=True,                # Enable logging for debugging
    engineio_logger=True        # Enable engine.io logging
)

print("[Extensions] ✅ SocketIO initialized")
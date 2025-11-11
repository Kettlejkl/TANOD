# run.py
"""
Flask application runner with Socket.IO support
"""

from app import app
from app.extensions import socketio

if __name__ == '__main__':
    # Run with socketio.run() instead of app.run()
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Important: disable reloader to prevent double initialization
    )
from flask import Flask
from flask_cors import CORS
from app.database import init_db
from app.extensions import socketio

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Initialize database
    init_db()

    # Import blueprints
    from app.api.analytics import analytics_bp, start_scheduler_thread
    from app.api.video import video_bp
    from app.api.auth.routes import auth_bp
    from app.api.alerts import alerts_bp  # ✅ ADD THIS LINE

    # Register Blueprints
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")
    app.register_blueprint(video_bp, url_prefix="/api/video")
    app.register_blueprint(auth_bp)  # already has /api/auth prefix
    app.register_blueprint(alerts_bp, url_prefix="/api/alerts")  # ✅ REGISTER ALERTS

    # Initialize SocketIO
    socketio.init_app(app)
    
    # Initialize cameras AFTER socketio is ready
    from app.api.video.routes import initialize_cameras
    initialize_cameras()

    # Start background scheduler
    start_scheduler_thread()

    return app

app = create_app()

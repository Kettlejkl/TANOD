# app/api/video/__init__.py
from .routes import video_bp
from .stream_manager import stream_manager

__all__ = ["video_bp", "stream_manager"]

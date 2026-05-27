from .stream_manager import VideoStreamManager
from .sync_manager import SynchronizedVideoManager
from .db_handler import AnalyticsDBHandler
from .alerts_sender import AlertsSender
from .video_processor import (
    hash_id,
    safe_get_track_confidence,
    convert_to_json_serializable,
    encode_frame,
    draw_tracking_box,
    draw_crowd_alert,
    draw_fire_alert,
    draw_smoke_alert,
    draw_sync_info,
    get_behavior_label,
    get_behavior_color,
    BOX_COLOR,
    LOITERING_COLOR,
    RUNNING_COLOR,
    VIOLENCE_COLOR,
    FALLEN_COLOR,
    FIRE_COLOR,
    SMOKE_COLOR,
    CROWD_COLOR
)

__version__ = '1.0.0'

__all__ = [
    'VideoStreamManager',
    'SynchronizedVideoManager',
    'AnalyticsDBHandler',
    'AlertsSender',
    'hash_id',
    'safe_get_track_confidence',
    'convert_to_json_serializable',
    'encode_frame',
    'draw_tracking_box',
    'draw_crowd_alert',
    'draw_fire_alert',
    'draw_smoke_alert',
    'draw_sync_info',
    'get_behavior_label',
    'get_behavior_color',
]

stream_manager = VideoStreamManager()
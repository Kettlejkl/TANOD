import cv2
import numpy as np
import base64
import time
from datetime import datetime
import hashlib

_global_uid_hash_map = {}

BOX_COLOR          = (0, 255, 0)
OUTSIDE_COLOR      = (0, 165, 255)
LOITERING_COLOR    = (0, 165, 255)
RUNNING_COLOR      = (0, 0, 255)
VIOLENCE_COLOR     = (128, 0, 128)
FALLEN_COLOR       = (255, 0, 255)
FIRE_COLOR         = (0, 69, 255)
SMOKE_COLOR        = (128, 128, 128)
CROWD_COLOR        = (0, 255, 255)
PENDING_COLOR      = (255, 165, 0)
INITIALIZING_COLOR = (128, 128, 128)


def hash_id(pid):
    global _global_uid_hash_map
    if pid in _global_uid_hash_map:
        return _global_uid_hash_map[pid]
    hash_str = hashlib.sha256(str(pid).encode()).hexdigest()[:8]
    _global_uid_hash_map[pid] = hash_str
    return hash_str


def safe_get_track_confidence(track, default=0.8):
    try:
        if hasattr(track, 'get_det_conf'):
            conf = track.get_det_conf()
            if conf is not None:
                conf_float = float(conf)
                return max(0.0, min(1.0, conf_float))
        return default
    except (TypeError, ValueError, AttributeError):
        return default


def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def encode_frame(frame, quality=65):
    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY,     quality,
        cv2.IMWRITE_JPEG_OPTIMIZE,    1,
        cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
    ]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    return base64.b64encode(buffer).decode('utf-8')


def encode_frame_fast(frame, quality=50):
    h, w = frame.shape[:2]
    if w > 640:
        scale = 640 / w
        frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY,     quality,
        cv2.IMWRITE_JPEG_OPTIMIZE,    0,
        cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
    ]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    return base64.b64encode(buffer).decode('utf-8')


def draw_tracking_box(frame, bbox, label, color):
    l, t, r, b = map(int, bbox)
    thickness      = 1
    font_scale     = 0.4
    font_thickness = 1
    cv2.rectangle(frame, (l, t), (r, b), color, thickness)
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(frame, (l, t - text_h - baseline - 6), (l + text_w + 4, t), color, -1)
    cv2.putText(frame, label, (l + 2, t - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                font_thickness, cv2.LINE_AA)


def draw_crowd_alert(frame, position, count):
    cv2.circle(frame, tuple(position), 80, CROWD_COLOR, 3)
    cv2.putText(frame, f"CROWD: {count} people",
                (position[0] - 80, position[1] - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CROWD_COLOR, 2)


def draw_crowd_hull(frame, hull_px, count, confidence=0.8):
    """
    Draws a filled + outlined bounding rect around a detected cluster,
    with a 'CLUSTER  N persons' label above the top-left corner.
    hull_px: [x1, y1, x2, y2]  (top-left / bottom-right of the group)
    """
    x1, y1, x2, y2 = hull_px

    # Clamp to frame bounds
    fh, fw = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(fw, x2); y2 = min(fh, y2)

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), CROWD_COLOR, -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # Border: thick black outline + thinner colored inner
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0),   4, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), CROWD_COLOR,  2, cv2.LINE_AA)

    # Corner tick marks for extra visual clarity
    tick = 14
    for (cx, cy, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * tick, cy),          CROWD_COLOR, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx,             cy + dy * tick), CROWD_COLOR, 2, cv2.LINE_AA)

    # Label
    label     = f"CLUSTER  {count} persons"
    font      = cv2.FONT_HERSHEY_DUPLEX
    scale     = 0.52
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 5

    lx1 = x1
    lx2 = x1 + tw + pad * 2
    ly2 = y1
    ly1 = max(0, y1 - th - pad * 2)

    bg = frame.copy()
    cv2.rectangle(bg, (lx1, ly1), (lx2, ly2), (20, 20, 20), -1)
    cv2.addWeighted(bg, 0.65, frame, 0.35, 0, frame)

    tx = lx1 + pad
    ty = ly2 - pad
    cv2.putText(frame, label, (tx + 1, ty + 1), font, scale, (0, 0, 0),
                thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (tx,     ty),     font, scale, CROWD_COLOR,
                thickness,     cv2.LINE_AA)


def draw_fire_alert(frame, position):
    cv2.circle(frame, tuple(position), 100, FIRE_COLOR, 4)
    cv2.putText(frame, "FIRE DETECTED",
                (position[0] - 100, position[1] - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, FIRE_COLOR, 3)


def draw_smoke_alert(frame, position):
    cv2.circle(frame, tuple(position), 100, SMOKE_COLOR, 4)
    cv2.putText(frame, "SMOKE DETECTED",
                (position[0] - 100, position[1] - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, SMOKE_COLOR, 3)


def draw_sync_info(frame, current_time, loop_count, use_sync):
    if use_sync:
        cv2.putText(frame, f"Master Time: {current_time:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Sync: ON | Loop #{loop_count}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    elif loop_count > 0:
        cv2.putText(frame, f"Loop #{loop_count}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def get_behavior_label(behavior_type, hashed_id, fence_name):
    labels = {
        'loitering': f"LOITERING UID {hashed_id} [IN:{fence_name}]",
        'running':   f"RUNNING UID {hashed_id} [IN:{fence_name}]",
        'violence':  f"VIOLENCE UID {hashed_id} [IN:{fence_name}]",
        'fallen':    f"FALLEN UID {hashed_id} [IN:{fence_name}]",
    }
    return labels.get(behavior_type, f"UID {hashed_id} [IN:{fence_name}]")


def get_behavior_color(behavior_type):
    colors = {
        'loitering': LOITERING_COLOR,
        'running':   RUNNING_COLOR,
        'violence':  VIOLENCE_COLOR,
        'fallen':    FALLEN_COLOR,
    }
    return colors.get(behavior_type, BOX_COLOR)
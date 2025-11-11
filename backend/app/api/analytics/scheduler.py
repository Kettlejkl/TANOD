# analytics/scheduler.py
"""
Periodically save analytics data to CSV files for 2-camera system
ONE ROW PER PERSON (not per frame)
"""

import time
from datetime import datetime, timedelta
from .exporter import exporter
from threading import Thread

EXPORT_INTERVAL_SECONDS = 10  # 1 hour for production
CAMERA_IDS = ["CAM001", "CAM002"]

# Global scheduler thread reference
_scheduler_thread = None
_scheduler_running = False

def run_periodic_exports():
    """Main scheduler loop"""
    global _scheduler_running
    _scheduler_running = True
    
    print("[Scheduler] 🚀 Scheduler started - exporting person-level summaries")
    print(f"[Scheduler] ⏱️  Export interval: {EXPORT_INTERVAL_SECONDS} seconds ({EXPORT_INTERVAL_SECONDS/3600:.1f} hours)")
    
    while _scheduler_running:
        print(f"[Scheduler] ⏳ Waiting {EXPORT_INTERVAL_SECONDS} seconds until next export...")
        time.sleep(EXPORT_INTERVAL_SECONDS)
        if _scheduler_running:  # Check again after sleep
            export_cycle()

def export_cycle():
    """Run one export cycle - person-level aggregation"""
    now = datetime.utcnow()
    start_time = now - timedelta(hours=1)  # Last 1 hour
    end_time = now

    print(f"\n{'='*80}")
    print(f"[Scheduler] 🔄 Export cycle at {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Scheduler] Time range: {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Use hour-based prefix for unique filenames
    hour_prefix = f"hourly_{now.strftime('%Y%m%d_%H00')}"
    
    # Export person summaries for each camera
    for camera_id in CAMERA_IDS:
        try:
            print(f"[Scheduler] 📊 Exporting {camera_id}...", end=" ")
            
            # Export person summary (ONE ROW PER PERSISTENT_ID)
            persons_file = exporter.export_persons_summary(
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                filename=f"{hour_prefix}_persons_{camera_id}.csv"
            )
            
            # Export behaviors by person
            behaviors_file = exporter.export_behaviors_by_person(
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                filename=f"{hour_prefix}_behaviors_{camera_id}.csv"
            )
            
            # Export hourly stats
            stats_file = exporter.export_hourly_stats(
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                filename=f"{hour_prefix}_stats_{camera_id}.csv"
            )
            
            # Count exported files
            files = [persons_file, behaviors_file, stats_file]
            exported = sum(1 for f in files if f is not None)
            
            print(f"✅ ({exported}/3 files)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Export cross-camera journeys (once per cycle)
    try:
        print(f"[Scheduler] 🚶 Exporting journeys...", end=" ")
        journey_file = exporter.export_person_journeys(
            start_time=start_time,
            end_time=end_time,
            filename=f"{hour_prefix}_journeys.csv"
        )
        print(f"✅" if journey_file else "⚠️  (no data)")
    except Exception as e:
        print(f"❌ {e}")

    print(f"\n{'='*80}")
    print(f"[Scheduler] ✅ Export cycle complete!")
    print(f"{'='*80}\n")

def start_scheduler():
    """Start the scheduler in a separate thread"""
    global _scheduler_thread, _scheduler_running
    
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        print("[Scheduler] ⚠️  Scheduler already running")
        return False
    
    _scheduler_running = True
    _scheduler_thread = Thread(target=run_periodic_exports, daemon=True, name="AnalyticsScheduler")
    _scheduler_thread.start()
    
    print(f"[Scheduler] ✅ Scheduler started for: {', '.join(CAMERA_IDS)}")
    print(f"[Scheduler] 📋 Exports: Person summaries (NOT per-frame)")
    return True

def stop_scheduler():
    """Stop the scheduler thread"""
    global _scheduler_running
    
    if not _scheduler_running:
        print("[Scheduler] ⚠️  Scheduler not running")
        return False
    
    print("[Scheduler] 🛑 Stopping scheduler...")
    _scheduler_running = False
    return True

def force_export():
    """Force an immediate export cycle"""
    print("[Scheduler] 🚀 Force export triggered")
    export_cycle()
    return True

def get_scheduler_status():
    """Get current scheduler status"""
    return {
        'running': _scheduler_running,
        'thread_alive': _scheduler_thread.is_alive() if _scheduler_thread else False,
        'export_interval': EXPORT_INTERVAL_SECONDS,
        'cameras': CAMERA_IDS
    }

# Legacy function name for backwards compatibility
def start_scheduler_thread():
    """Alias for start_scheduler() - for backwards compatibility"""
    return start_scheduler()

# Module-level exports
__all__ = [
    'start_scheduler',
    'stop_scheduler',
    'force_export',
    'get_scheduler_status',
    'start_scheduler_thread',  # Legacy
    'export_cycle',
    'EXPORT_INTERVAL_SECONDS',
    'CAMERA_IDS'
]
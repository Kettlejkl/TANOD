"""
Periodically save analytics data to CSV files for 2-camera system
ONE ROW PER PERSON (not per frame)
"""

import time
from datetime import datetime, timedelta
from threading import Thread
from .exporter import exporter

EXPORT_INTERVAL_SECONDS = 10  # Set to 3600 for production (1 hour)
CAMERA_IDS = ["CAM001", "CAM002"]

# -------------------------------------------------------------------------
# Core Export Logic
# -------------------------------------------------------------------------

def export_cycle():
    """Run one export cycle - person-level aggregation"""
    now = datetime.utcnow()
    start_time = now - timedelta(hours=1)
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

            persons_file = exporter.export_persons_summary(
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                filename=f"{hour_prefix}_persons_{camera_id}.csv"
            )

            behaviors_file = exporter.export_behaviors_by_person(
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                filename=f"{hour_prefix}_behaviors_{camera_id}.csv"
            )

            stats_file = exporter.export_hourly_stats(
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                filename=f"{hour_prefix}_stats_{camera_id}.csv"
            )

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

# -------------------------------------------------------------------------
# Periodic Thread Runner
# -------------------------------------------------------------------------

def run_periodic_exports():
    """Continuously run exports every EXPORT_INTERVAL_SECONDS."""
    print("[Scheduler] 🚀 Scheduler started - exporting person-level summaries")
    print(f"[Scheduler] ⏱️  Export interval: {EXPORT_INTERVAL_SECONDS} seconds "
          f"({EXPORT_INTERVAL_SECONDS / 3600:.1f} hours)")

    while True:
        print(f"[Scheduler] ⏳ Waiting {EXPORT_INTERVAL_SECONDS} seconds until next export...")
        time.sleep(EXPORT_INTERVAL_SECONDS)
        export_cycle()

def start_scheduler_thread():
    """Run the scheduler in a separate daemon thread."""
    t = Thread(target=run_periodic_exports, daemon=True, name="AnalyticsScheduler")
    t.start()
    print(f"[Scheduler] ✅ Scheduler thread started for: {', '.join(CAMERA_IDS)}")
    print(f"[Scheduler] 📋 Exports: Person summaries (NOT per-frame)")

# -------------------------------------------------------------------------
# Compatibility Aliases (for __init__.py)
# -------------------------------------------------------------------------

# For backward compatibility with older imports
scheduler = None  # Placeholder for compatibility (no global APScheduler instance)

def start_scheduler():
    """Start the analytics export scheduler."""
    start_scheduler_thread()

def stop_scheduler():
    """Stop the scheduler (not implemented for thread-based version)."""
    print("[Scheduler] ⚠️ Stop not implemented in this thread-based scheduler")

def force_export():
    """Force an immediate export cycle."""
    export_cycle()

def get_scheduler_status():
    """Return current scheduler status."""
    return {
        "status": "running",
        "interval_seconds": EXPORT_INTERVAL_SECONDS,
        "cameras": CAMERA_IDS,
    }

# -------------------------------------------------------------------------
# End of File
# -------------------------------------------------------------------------

# app/api/analytics/migrate_database.py
"""
Database migration script to update schema for new behavior types
Adds: violence_events, fallen_events, fire_events, smoke_events
Removes: suspicious_events (deprecated)
"""

import sqlite3
import os
from datetime import datetime


def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def migrate_camera_statistics(db_path='analytics.db'):
    """
    Migrate camera_statistics table
    Add: violence_events, fallen_events, fire_events, smoke_events
    """
    print("\n" + "="*80)
    print("MIGRATING camera_statistics TABLE")
    print("="*80)
    
    if not os.path.exists(db_path):
        print(f"⚠️  Database not found: {db_path}")
        print("   Creating new database with updated schema...")
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check and add new columns
        new_columns = [
            ('violence_events', 'INTEGER DEFAULT 0'),
            ('fallen_events', 'INTEGER DEFAULT 0'),
            ('fire_events', 'INTEGER DEFAULT 0'),
            ('smoke_events', 'INTEGER DEFAULT 0')
        ]
        
        for col_name, col_type in new_columns:
            if not check_column_exists(cursor, 'camera_statistics', col_name):
                print(f"➕ Adding column: {col_name}")
                cursor.execute(f"ALTER TABLE camera_statistics ADD COLUMN {col_name} {col_type}")
                conn.commit()
            else:
                print(f"✅ Column already exists: {col_name}")
        
        # Check for deprecated column
        if check_column_exists(cursor, 'camera_statistics', 'suspicious_events'):
            print("⚠️  Found deprecated column: suspicious_events")
            print("   (Will be ignored in queries but kept for historical data)")
        
        print("✅ camera_statistics migration complete")
        return True
        
    except Exception as e:
        print(f"❌ Error migrating camera_statistics: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def migrate_daily_reports(db_path='analytics.db'):
    """
    Migrate daily_reports table
    Add: total_violence, total_fallen, total_fire, total_smoke
    """
    print("\n" + "="*80)
    print("MIGRATING daily_reports TABLE")
    print("="*80)
    
    if not os.path.exists(db_path):
        print(f"⚠️  Database not found: {db_path}")
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check and add new columns
        new_columns = [
            ('total_violence', 'INTEGER DEFAULT 0'),
            ('total_fallen', 'INTEGER DEFAULT 0'),
            ('total_fire', 'INTEGER DEFAULT 0'),
            ('total_smoke', 'INTEGER DEFAULT 0')
        ]
        
        for col_name, col_type in new_columns:
            if not check_column_exists(cursor, 'daily_reports', col_name):
                print(f"➕ Adding column: {col_name}")
                cursor.execute(f"ALTER TABLE daily_reports ADD COLUMN {col_name} {col_type}")
                conn.commit()
            else:
                print(f"✅ Column already exists: {col_name}")
        
        # Check for deprecated column
        if check_column_exists(cursor, 'daily_reports', 'total_suspicious'):
            print("⚠️  Found deprecated column: total_suspicious")
            print("   (Will be ignored in queries but kept for historical data)")
        
        print("✅ daily_reports migration complete")
        return True
        
    except Exception as e:
        print(f"❌ Error migrating daily_reports: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def migrate_behavior_events(db_path='analytics.db'):
    """
    Migrate behavior_events table
    Add: detection_method column (for transparency)
    Make: detection_id nullable (for non-person events like fire/smoke)
    """
    print("\n" + "="*80)
    print("MIGRATING behavior_events TABLE")
    print("="*80)
    
    if not os.path.exists(db_path):
        print(f"⚠️  Database not found: {db_path}")
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Add detection_method column
        if not check_column_exists(cursor, 'behavior_events', 'detection_method'):
            print("➕ Adding column: detection_method")
            cursor.execute("ALTER TABLE behavior_events ADD COLUMN detection_method VARCHAR(50)")
            conn.commit()
        else:
            print("✅ Column already exists: detection_method")
        
        # Note: SQLite doesn't support making columns nullable after creation
        # The models.py already has detection_id as nullable=True
        print("ℹ️  detection_id nullability handled in models.py")
        
        # Update existing records with default detection method
        cursor.execute("""
            UPDATE behavior_events 
            SET detection_method = 
                CASE behavior_type
                    WHEN 'violence' THEN 'yolo_pose'
                    WHEN 'fire' THEN 'yolo_object_detection'
                    WHEN 'smoke' THEN 'yolo_object_detection'
                    WHEN 'fallen' THEN 'movement_analysis'
                    ELSE 'movement_analysis'
                END
            WHERE detection_method IS NULL
        """)
        conn.commit()
        print("✅ Updated detection_method for existing records")
        
        print("✅ behavior_events migration complete")
        return True
        
    except Exception as e:
        print(f"❌ Error migrating behavior_events: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def verify_migration(db_path='analytics.db'):
    """Verify migration was successful"""
    print("\n" + "="*80)
    print("VERIFYING MIGRATION")
    print("="*80)
    
    if not os.path.exists(db_path):
        print("⚠️  Database not found - will be created on first use")
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check camera_statistics
        cursor.execute("PRAGMA table_info(camera_statistics)")
        stats_columns = {row[1] for row in cursor.fetchall()}
        
        required_stats_cols = {
            'violence_events', 'fallen_events', 'fire_events', 'smoke_events'
        }
        
        if required_stats_cols.issubset(stats_columns):
            print("✅ camera_statistics has all required columns")
        else:
            missing = required_stats_cols - stats_columns
            print(f"❌ camera_statistics missing columns: {missing}")
            return False
        
        # Check daily_reports
        cursor.execute("PRAGMA table_info(daily_reports)")
        reports_columns = {row[1] for row in cursor.fetchall()}
        
        required_reports_cols = {
            'total_violence', 'total_fallen', 'total_fire', 'total_smoke'
        }
        
        if required_reports_cols.issubset(reports_columns):
            print("✅ daily_reports has all required columns")
        else:
            missing = required_reports_cols - reports_columns
            print(f"❌ daily_reports missing columns: {missing}")
            return False
        
        # Check behavior_events
        cursor.execute("PRAGMA table_info(behavior_events)")
        behavior_columns = {row[1] for row in cursor.fetchall()}
        
        if 'detection_method' in behavior_columns:
            print("✅ behavior_events has detection_method column")
        else:
            print("❌ behavior_events missing detection_method column")
            return False
        
        # Check for new behavior types in data
        cursor.execute("""
            SELECT DISTINCT behavior_type 
            FROM behavior_events 
            WHERE behavior_type IN ('violence', 'fallen', 'fire', 'smoke')
        """)
        new_types = [row[0] for row in cursor.fetchall()]
        
        if new_types:
            print(f"✅ Found new behavior types in data: {', '.join(new_types)}")
        else:
            print("ℹ️  No new behavior types in data yet (expected for new installations)")
        
        print("\n✅ MIGRATION VERIFICATION PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    finally:
        conn.close()


def create_backup(db_path='analytics.db'):
    """Create a backup of the database before migration"""
    if not os.path.exists(db_path):
        print("ℹ️  No database to backup")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{db_path}.backup_{timestamp}"
    
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"⚠️  Backup failed: {e}")
        return None


def run_migration(db_path='analytics.db', create_backup_first=True):
    """
    Run complete migration
    
    Args:
        db_path: Path to SQLite database
        create_backup_first: Whether to create backup before migration
    
    Returns:
        bool: Success status
    """
    print("\n" + "🔄 "*40)
    print("ANALYTICS DATABASE MIGRATION")
    print("🔄 "*40)
    print(f"\nDatabase: {db_path}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create backup
    if create_backup_first:
        print("\n📦 Creating backup...")
        backup_path = create_backup(db_path)
        if backup_path:
            print(f"   Backup: {backup_path}")
    
    # Run migrations
    success = True
    success &= migrate_camera_statistics(db_path)
    success &= migrate_daily_reports(db_path)
    success &= migrate_behavior_events(db_path)
    
    # Verify
    success &= verify_migration(db_path)
    
    if success:
        print("\n" + "✅ "*40)
        print("MIGRATION COMPLETED SUCCESSFULLY")
        print("✅ "*40)
        print("\nNew behavior types supported:")
        print("  ✅ violence (YOLO-Pose based)")
        print("  ✅ fallen (movement + aspect ratio)")
        print("  ✅ fire (YOLO object detection)")
        print("  ✅ smoke (YOLO object detection)")
        print("\nDeprecated types (removed from new detections):")
        print("  ⚠️  suspicious (historical data preserved)")
    else:
        print("\n" + "❌ "*40)
        print("MIGRATION FAILED")
        print("❌ "*40)
        if backup_path:
            print(f"\n⚠️  Restore from backup if needed: {backup_path}")
    
    return success


if __name__ == "__main__":
    # Run migration
    run_migration('analytics.db', create_backup_first=True)
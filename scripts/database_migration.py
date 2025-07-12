#!/usr/bin/env python3
"""
Database Migration Script
Fixes critical database schema issues and migrates existing data
"""

import asyncio
import asyncpg
import logging
import sys
import time
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handle database migrations and schema fixes"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        
    async def connect(self):
        """Connect to database"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=3,
                command_timeout=60
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            logger.info("✅ Connected to database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    async def backup_problematic_data(self) -> Dict[str, Any]:
        """Backup data that might be affected by schema changes"""
        backup_data = {}
        
        try:
            async with self.pool.acquire() as conn:
                # Backup shots with string shot_ids (problematic ones)
                string_shots = await conn.fetch("""
                    SELECT * FROM shots 
                    WHERE shot_id::text ~ '^[a-zA-Z]' 
                    ORDER BY video_id, shot_index
                """)
                backup_data['string_shots'] = [dict(row) for row in string_shots]
                
                # Backup keyframes that reference string shot_ids
                if backup_data['string_shots']:
                    string_shot_ids = [shot['shot_id'] for shot in backup_data['string_shots']]
                    affected_keyframes = await conn.fetch("""
                        SELECT * FROM keyframes 
                        WHERE shot_id = ANY($1::int[])
                    """, string_shot_ids)
                    backup_data['affected_keyframes'] = [dict(row) for row in affected_keyframes]
                else:
                    backup_data['affected_keyframes'] = []
                
                logger.info(f"📦 Backed up {len(backup_data['string_shots'])} shots and {len(backup_data['affected_keyframes'])} keyframes")
                
        except Exception as e:
            logger.warning(f"⚠️  Backup failed (continuing anyway): {e}")
            backup_data = {'string_shots': [], 'affected_keyframes': []}
        
        return backup_data
    
    async def fix_schema_issues(self) -> Dict[str, Any]:
        """Fix critical schema issues"""
        results = {
            'embeddings_unique_constraint': False,
            'shot_id_cleanup': False,
            'foreign_key_cleanup': False
        }
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # 1. Add unique constraint to embeddings table if missing
                    try:
                        await conn.execute("""
                            ALTER TABLE embeddings 
                            DROP CONSTRAINT IF EXISTS embeddings_keyframe_id_model_name_key
                        """)
                        await conn.execute("""
                            ALTER TABLE embeddings 
                            ADD CONSTRAINT embeddings_keyframe_id_model_name_key 
                            UNIQUE (keyframe_id, model_name)
                        """)
                        results['embeddings_unique_constraint'] = True
                        logger.info("✅ Fixed embeddings unique constraint")
                    except Exception as e:
                        logger.warning(f"⚠️  Embeddings constraint fix failed: {e}")
                    
                    # 2. Clean up orphaned records
                    deleted_embeddings = await conn.fetchval("""
                        DELETE FROM embeddings 
                        WHERE keyframe_id NOT IN (SELECT keyframe_id FROM keyframes)
                    """)
                    
                    deleted_keyframes = await conn.fetchval("""
                        DELETE FROM keyframes 
                        WHERE shot_id NOT IN (SELECT shot_id FROM shots)
                    """)
                    
                    deleted_shots = await conn.fetchval("""
                        DELETE FROM shots 
                        WHERE video_id NOT IN (SELECT video_id FROM videos)
                    """)
                    
                    results['foreign_key_cleanup'] = True
                    logger.info(f"✅ Cleaned up {deleted_embeddings or 0} orphaned embeddings, "
                              f"{deleted_keyframes or 0} orphaned keyframes, "
                              f"{deleted_shots or 0} orphaned shots")
                    
        except Exception as e:
            logger.error(f"❌ Schema fix failed: {e}")
            results['error'] = str(e)
        
        return results
    
    async def migrate_shot_references(self, backup_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate shot references from string IDs to integer IDs"""
        results = {
            'shots_migrated': 0,
            'keyframes_updated': 0,
            'errors': []
        }
        
        if not backup_data.get('string_shots'):
            logger.info("ℹ️  No string shot IDs found - migration not needed")
            return results
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Group shots by video for proper handling
                    shots_by_video = {}
                    for shot in backup_data['string_shots']:
                        video_id = shot['video_id']
                        if video_id not in shots_by_video:
                            shots_by_video[video_id] = []
                        shots_by_video[video_id].append(shot)
                    
                    for video_id, shots in shots_by_video.items():
                        try:
                            # Delete old string-based shots for this video
                            string_shot_ids = [shot['shot_id'] for shot in shots]
                            await conn.execute("""
                                DELETE FROM shots WHERE shot_id = ANY($1::text[])
                            """, string_shot_ids)
                            
                            # Re-insert with proper integer shot_ids
                            shot_id_mapping = {}
                            for shot in sorted(shots, key=lambda x: x['shot_index']):
                                new_shot_id = await conn.fetchval("""
                                    INSERT INTO shots (
                                        video_id, shot_index, start_frame, end_frame, 
                                        start_time, end_time, confidence, created_at
                                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                                    RETURNING shot_id
                                """, 
                                    shot['video_id'], shot['shot_index'], shot['start_frame'],
                                    shot['end_frame'], shot['start_time'], shot['end_time'],
                                    shot.get('confidence', 0.0), shot.get('created_at'))
                                
                                shot_id_mapping[shot['shot_id']] = new_shot_id
                                results['shots_migrated'] += 1
                            
                            # Update keyframes to use new shot_ids
                            for keyframe in backup_data['affected_keyframes']:
                                if keyframe['video_id'] == video_id:
                                    old_shot_id = keyframe['shot_id']
                                    if old_shot_id in shot_id_mapping:
                                        new_shot_id = shot_id_mapping[old_shot_id]
                                        await conn.execute("""
                                            UPDATE keyframes 
                                            SET shot_id = $1 
                                            WHERE keyframe_id = $2
                                        """, new_shot_id, keyframe['keyframe_id'])
                                        results['keyframes_updated'] += 1
                            
                            logger.info(f"✅ Migrated {len(shots)} shots for video {video_id}")
                            
                        except Exception as e:
                            error_msg = f"Failed to migrate video {video_id}: {e}"
                            logger.error(f"❌ {error_msg}")
                            results['errors'].append(error_msg)
        
        except Exception as e:
            error_msg = f"Migration transaction failed: {e}"
            logger.error(f"❌ {error_msg}")
            results['errors'].append(error_msg)
        
        return results
    
    async def verify_migration(self) -> Dict[str, Any]:
        """Verify migration was successful"""
        verification = {
            'foreign_key_violations': 0,
            'orphaned_records': 0,
            'schema_valid': True,
            'errors': []
        }
        
        try:
            async with self.pool.acquire() as conn:
                # Check for foreign key violations
                orphaned_keyframes = await conn.fetchval("""
                    SELECT COUNT(*) FROM keyframes k 
                    LEFT JOIN shots s ON k.shot_id = s.shot_id 
                    WHERE s.shot_id IS NULL
                """)
                verification['orphaned_records'] += orphaned_keyframes or 0
                
                orphaned_embeddings = await conn.fetchval("""
                    SELECT COUNT(*) FROM embeddings e 
                    LEFT JOIN keyframes k ON e.keyframe_id = k.keyframe_id 
                    WHERE k.keyframe_id IS NULL
                """)
                verification['orphaned_records'] += orphaned_embeddings or 0
                
                # Check unique constraints
                duplicate_embeddings = await conn.fetchval("""
                    SELECT COUNT(*) FROM (
                        SELECT keyframe_id, model_name, COUNT(*) 
                        FROM embeddings 
                        GROUP BY keyframe_id, model_name 
                        HAVING COUNT(*) > 1
                    ) duplicates
                """)
                
                if duplicate_embeddings and duplicate_embeddings > 0:
                    verification['schema_valid'] = False
                    verification['errors'].append(f"Found {duplicate_embeddings} duplicate embeddings")
                
        except Exception as e:
            verification['errors'].append(f"Verification failed: {e}")
            verification['schema_valid'] = False
        
        return verification
    
    async def run_migration(self) -> Dict[str, Any]:
        """Run complete migration process"""
        migration_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "connection_status": "unknown",
            "backup_data": {},
            "schema_fixes": {},
            "migration_results": {},
            "verification": {},
            "success": False
        }
        
        # Connect to database
        if not await self.connect():
            migration_report["connection_status"] = "failed"
            return migration_report
        
        migration_report["connection_status"] = "connected"
        
        logger.info("🚀 Starting database migration...")
        
        # Step 1: Backup problematic data
        logger.info("📦 Backing up data...")
        migration_report["backup_data"] = await self.backup_problematic_data()
        
        # Step 2: Fix schema issues
        logger.info("🔧 Fixing schema issues...")
        migration_report["schema_fixes"] = await self.fix_schema_issues()
        
        # Step 3: Migrate shot references
        logger.info("🔄 Migrating shot references...")
        migration_report["migration_results"] = await self.migrate_shot_references(
            migration_report["backup_data"]
        )
        
        # Step 4: Verify migration
        logger.info("✅ Verifying migration...")
        migration_report["verification"] = await self.verify_migration()
        
        # Determine overall success
        migration_report["success"] = (
            migration_report["verification"]["schema_valid"] and
            migration_report["verification"]["orphaned_records"] == 0 and
            len(migration_report["verification"]["errors"]) == 0
        )
        
        return migration_report
    
    async def disconnect(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()

async def main():
    """Main entry point for database migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Migration Tool")
    parser.add_argument("--database-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No changes will be made")
        # TODO: Implement dry run logic
        return
    
    migrator = DatabaseMigrator(args.database_url)
    
    try:
        report = await migrator.run_migration()
        print_migration_report(report)
        
        if report["success"]:
            logger.info("🎉 Migration completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Migration failed or incomplete")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Migration stopped by user")
        sys.exit(1)
    finally:
        await migrator.disconnect()

def print_migration_report(report: Dict[str, Any]):
    """Print formatted migration report"""
    print("\n" + "="*60)
    print(f"🔧 DATABASE MIGRATION REPORT - {report['timestamp']}")
    print("="*60)
    
    # Connection
    print(f"🔗 Connection: {report['connection_status']}")
    
    # Backup
    backup = report.get('backup_data', {})
    print(f"📦 Backup: {len(backup.get('string_shots', []))} shots, {len(backup.get('affected_keyframes', []))} keyframes")
    
    # Schema fixes
    schema_fixes = report.get('schema_fixes', {})
    print(f"🔧 Schema Fixes:")
    print(f"  - Embeddings constraint: {'✅' if schema_fixes.get('embeddings_unique_constraint') else '❌'}")
    print(f"  - Foreign key cleanup: {'✅' if schema_fixes.get('foreign_key_cleanup') else '❌'}")
    
    # Migration results
    migration = report.get('migration_results', {})
    print(f"🔄 Migration:")
    print(f"  - Shots migrated: {migration.get('shots_migrated', 0)}")
    print(f"  - Keyframes updated: {migration.get('keyframes_updated', 0)}")
    if migration.get('errors'):
        print(f"  - Errors: {len(migration['errors'])}")
    
    # Verification
    verification = report.get('verification', {})
    print(f"✅ Verification:")
    print(f"  - Orphaned records: {verification.get('orphaned_records', 'unknown')}")
    print(f"  - Schema valid: {'✅' if verification.get('schema_valid') else '❌'}")
    if verification.get('errors'):
        print(f"  - Issues: {len(verification['errors'])}")
    
    # Overall status
    success_icon = "✅" if report.get('success') else "❌"
    success_text = "SUCCESS" if report.get('success') else "FAILED"
    print(f"\n{success_icon} Overall Status: {success_text}")

if __name__ == "__main__":
    asyncio.run(main())

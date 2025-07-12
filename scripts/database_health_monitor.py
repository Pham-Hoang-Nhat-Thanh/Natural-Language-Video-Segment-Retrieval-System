#!/usr/bin/env python3
"""
Database Health Monitor and Issue Resolver
Monitors database connections, schema consistency, and performs automated repairs
"""

import asyncio
import asyncpg
import logging
import sys
import time
from typing import Dict, List, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseHealthMonitor:
    """Monitor and fix database health issues"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        
    async def connect(self):
        """Connect to database with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=1,
                    max_size=5,
                    command_timeout=30
                )
                
                # Test connection
                async with self.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                
                logger.info(f"✅ Connected to database on attempt {attempt + 1}")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️  Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
        
        logger.error("❌ Failed to connect to database")
        return False
    
    async def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = $1
                    )
                """, table_name)
                return bool(result)
        except Exception as e:
            logger.error(f"Error checking table {table_name}: {e}")
            return False
    
    async def check_foreign_key_constraints(self) -> List[Dict[str, Any]]:
        """Check for foreign key constraint violations"""
        issues = []
        
        try:
            async with self.pool.acquire() as conn:
                # Check shots -> videos relationship
                orphaned_shots = await conn.fetch("""
                    SELECT s.shot_id, s.video_id 
                    FROM shots s 
                    LEFT JOIN videos v ON s.video_id = v.video_id 
                    WHERE v.video_id IS NULL
                """)
                
                if orphaned_shots:
                    issues.append({
                        "type": "orphaned_shots",
                        "count": len(orphaned_shots),
                        "description": "Shots with no corresponding video",
                        "data": [dict(row) for row in orphaned_shots[:5]]  # First 5 examples
                    })
                
                # Check keyframes -> shots relationship
                orphaned_keyframes = await conn.fetch("""
                    SELECT k.keyframe_id, k.shot_id 
                    FROM keyframes k 
                    LEFT JOIN shots s ON k.shot_id = s.shot_id 
                    WHERE s.shot_id IS NULL
                """)
                
                if orphaned_keyframes:
                    issues.append({
                        "type": "orphaned_keyframes", 
                        "count": len(orphaned_keyframes),
                        "description": "Keyframes with no corresponding shot",
                        "data": [dict(row) for row in orphaned_keyframes[:5]]
                    })
                
                # Check embeddings -> keyframes relationship
                orphaned_embeddings = await conn.fetch("""
                    SELECT e.embedding_id, e.keyframe_id 
                    FROM embeddings e 
                    LEFT JOIN keyframes k ON e.keyframe_id = k.keyframe_id 
                    WHERE k.keyframe_id IS NULL
                """)
                
                if orphaned_embeddings:
                    issues.append({
                        "type": "orphaned_embeddings",
                        "count": len(orphaned_embeddings),
                        "description": "Embeddings with no corresponding keyframe",
                        "data": [dict(row) for row in orphaned_embeddings[:5]]
                    })
                
        except Exception as e:
            logger.error(f"Error checking foreign key constraints: {e}")
            issues.append({
                "type": "constraint_check_error",
                "description": str(e)
            })
        
        return issues
    
    async def check_data_consistency(self) -> List[Dict[str, Any]]:
        """Check for data consistency issues"""
        issues = []
        
        try:
            async with self.pool.acquire() as conn:
                # Check for videos with no shots
                videos_no_shots = await conn.fetch("""
                    SELECT v.video_id, v.status 
                    FROM videos v 
                    LEFT JOIN shots s ON v.video_id = s.video_id 
                    WHERE s.video_id IS NULL AND v.status = 'completed'
                """)
                
                if videos_no_shots:
                    issues.append({
                        "type": "videos_no_shots",
                        "count": len(videos_no_shots),
                        "description": "Completed videos with no shots",
                        "data": [dict(row) for row in videos_no_shots[:5]]
                    })
                
                # Check for shots with no keyframes
                shots_no_keyframes = await conn.fetch("""
                    SELECT s.shot_id, s.video_id 
                    FROM shots s 
                    LEFT JOIN keyframes k ON s.shot_id = k.shot_id 
                    WHERE k.shot_id IS NULL
                """)
                
                if shots_no_keyframes:
                    issues.append({
                        "type": "shots_no_keyframes",
                        "count": len(shots_no_keyframes),
                        "description": "Shots with no keyframes",
                        "data": [dict(row) for row in shots_no_keyframes[:5]]
                    })
                
                # Check for keyframes with no embeddings
                keyframes_no_embeddings = await conn.fetch("""
                    SELECT k.keyframe_id, k.video_id 
                    FROM keyframes k 
                    LEFT JOIN embeddings e ON k.keyframe_id = e.keyframe_id 
                    WHERE e.keyframe_id IS NULL
                """)
                
                if keyframes_no_embeddings:
                    issues.append({
                        "type": "keyframes_no_embeddings",
                        "count": len(keyframes_no_embeddings),
                        "description": "Keyframes with no embeddings",
                        "data": [dict(row) for row in keyframes_no_embeddings[:5]]
                    })
                
        except Exception as e:
            logger.error(f"Error checking data consistency: {e}")
            issues.append({
                "type": "consistency_check_error",
                "description": str(e)
            })
        
        return issues
    
    async def fix_orphaned_records(self) -> Dict[str, int]:
        """Fix orphaned records by cleaning them up"""
        fixed_counts = {}
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Clean up orphaned embeddings
                    deleted_embeddings = await conn.fetchval("""
                        DELETE FROM embeddings 
                        WHERE keyframe_id NOT IN (SELECT keyframe_id FROM keyframes)
                        RETURNING COUNT(*)
                    """)
                    fixed_counts['orphaned_embeddings'] = deleted_embeddings or 0
                    
                    # Clean up orphaned keyframes
                    deleted_keyframes = await conn.fetchval("""
                        DELETE FROM keyframes 
                        WHERE shot_id NOT IN (SELECT shot_id FROM shots)
                        RETURNING COUNT(*)
                    """)
                    fixed_counts['orphaned_keyframes'] = deleted_keyframes or 0
                    
                    # Clean up orphaned shots
                    deleted_shots = await conn.fetchval("""
                        DELETE FROM shots 
                        WHERE video_id NOT IN (SELECT video_id FROM videos)
                        RETURNING COUNT(*)
                    """)
                    fixed_counts['orphaned_shots'] = deleted_shots or 0
                    
        except Exception as e:
            logger.error(f"Error fixing orphaned records: {e}")
            fixed_counts['error'] = str(e)
        
        return fixed_counts
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        try:
            async with self.pool.acquire() as conn:
                # Table counts
                stats['videos'] = await conn.fetchval("SELECT COUNT(*) FROM videos")
                stats['shots'] = await conn.fetchval("SELECT COUNT(*) FROM shots")
                stats['keyframes'] = await conn.fetchval("SELECT COUNT(*) FROM keyframes")
                stats['embeddings'] = await conn.fetchval("SELECT COUNT(*) FROM embeddings")
                
                # Status breakdown
                video_statuses = await conn.fetch("""
                    SELECT status, COUNT(*) as count 
                    FROM videos 
                    GROUP BY status
                """)
                stats['video_statuses'] = {row['status']: row['count'] for row in video_statuses}
                
                # Database size
                db_size = await conn.fetchval("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """)
                stats['database_size'] = db_size
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    async def run_health_check(self, fix_issues: bool = False) -> Dict[str, Any]:
        """Run comprehensive health check"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "connection_status": "unknown",
            "tables_exist": {},
            "foreign_key_issues": [],
            "consistency_issues": [],
            "database_stats": {},
            "fixes_applied": {}
        }
        
        # Test connection
        if await self.connect():
            report["connection_status"] = "healthy"
        else:
            report["connection_status"] = "failed"
            return report
        
        # Check tables exist
        tables = ["videos", "shots", "keyframes", "embeddings", "processing_stats"]
        for table in tables:
            report["tables_exist"][table] = await self.check_table_exists(table)
        
        # Check foreign key constraints
        report["foreign_key_issues"] = await self.check_foreign_key_constraints()
        
        # Check data consistency
        report["consistency_issues"] = await self.check_data_consistency()
        
        # Get database stats
        report["database_stats"] = await self.get_database_stats()
        
        # Fix issues if requested
        if fix_issues and (report["foreign_key_issues"] or report["consistency_issues"]):
            logger.info("🔧 Fixing database issues...")
            report["fixes_applied"] = await self.fix_orphaned_records()
        
        return report
    
    async def disconnect(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()

async def main():
    """Main entry point for database health check"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Health Monitor")
    parser.add_argument("--database-url", required=True, help="PostgreSQL connection URL")
    parser.add_argument("--fix", action="store_true", help="Fix found issues automatically")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    monitor = DatabaseHealthMonitor(args.database_url)
    
    try:
        if args.watch:
            logger.info(f"🔍 Starting continuous monitoring (interval: {args.interval}s)")
            while True:
                report = await monitor.run_health_check(fix_issues=args.fix)
                print_health_report(report)
                await asyncio.sleep(args.interval)
        else:
            logger.info("🔍 Running one-time health check...")
            report = await monitor.run_health_check(fix_issues=args.fix)
            print_health_report(report)
            
    except KeyboardInterrupt:
        logger.info("Health monitoring stopped by user")
    finally:
        await monitor.disconnect()

def print_health_report(report: Dict[str, Any]):
    """Print formatted health report"""
    print("\n" + "="*50)
    print(f"📊 DATABASE HEALTH REPORT - {report['timestamp']}")
    print("="*50)
    
    # Connection status
    status_icon = "✅" if report["connection_status"] == "healthy" else "❌"
    print(f"{status_icon} Connection Status: {report['connection_status']}")
    
    # Tables
    print("\n📋 Tables:")
    for table, exists in report["tables_exist"].items():
        icon = "✅" if exists else "❌"
        print(f"  {icon} {table}")
    
    # Statistics
    if report["database_stats"]:
        stats = report["database_stats"]
        print(f"\n📈 Statistics:")
        print(f"  📦 Database Size: {stats.get('database_size', 'unknown')}")
        print(f"  🎥 Videos: {stats.get('videos', 0)}")
        print(f"  🎬 Shots: {stats.get('shots', 0)}")
        print(f"  🖼️  Keyframes: {stats.get('keyframes', 0)}")
        print(f"  🔗 Embeddings: {stats.get('embeddings', 0)}")
        
        if 'video_statuses' in stats:
            print(f"  📊 Video Status:")
            for status, count in stats['video_statuses'].items():
                print(f"    - {status}: {count}")
    
    # Issues
    total_fk_issues = sum(issue.get('count', 1) for issue in report["foreign_key_issues"])
    total_consistency_issues = sum(issue.get('count', 1) for issue in report["consistency_issues"])
    
    if total_fk_issues > 0:
        print(f"\n⚠️  Foreign Key Issues: {total_fk_issues}")
        for issue in report["foreign_key_issues"][:3]:  # Show first 3
            print(f"  - {issue['description']}: {issue.get('count', 1)}")
    
    if total_consistency_issues > 0:
        print(f"\n⚠️  Consistency Issues: {total_consistency_issues}")
        for issue in report["consistency_issues"][:3]:  # Show first 3
            print(f"  - {issue['description']}: {issue.get('count', 1)}")
    
    # Fixes applied
    if report["fixes_applied"]:
        print(f"\n🔧 Fixes Applied:")
        for fix_type, count in report["fixes_applied"].items():
            if fix_type != 'error' and count > 0:
                print(f"  - {fix_type}: {count} records cleaned")
    
    # Overall health
    is_healthy = (report["connection_status"] == "healthy" and 
                 all(report["tables_exist"].values()) and
                 total_fk_issues == 0 and total_consistency_issues == 0)
    
    health_icon = "✅" if is_healthy else "⚠️" 
    health_status = "HEALTHY" if is_healthy else "NEEDS ATTENTION"
    print(f"\n{health_icon} Overall Status: {health_status}")

if __name__ == "__main__":
    asyncio.run(main())

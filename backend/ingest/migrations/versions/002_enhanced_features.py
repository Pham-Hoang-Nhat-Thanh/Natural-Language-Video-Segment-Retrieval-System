"""Enhanced schema for multi-modal features support

Revision ID: 002
Revises: 001
Create Date: 2025-07-15 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    # Create enhanced_features table for multi-modal feature storage
    op.create_table('enhanced_features',
        sa.Column('feature_id', sa.String(), nullable=False),
        sa.Column('keyframe_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        
        # Core embeddings
        sa.Column('clip_embedding', sa.LargeBinary(), nullable=True),
        sa.Column('combined_embedding', sa.LargeBinary(), nullable=True),
        
        # Feature vectors and metadata
        sa.Column('object_features', postgresql.JSONB(), nullable=True),
        sa.Column('scene_features', postgresql.JSONB(), nullable=True),
        sa.Column('action_features', postgresql.JSONB(), nullable=True),
        sa.Column('text_features', postgresql.JSONB(), nullable=True),
        sa.Column('color_features', postgresql.JSONB(), nullable=True),
        
        # Searchable indices for quick filtering
        sa.Column('objects_detected', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('scene_type', sa.String(), nullable=True),
        sa.Column('primary_action', sa.String(), nullable=True),
        sa.Column('has_text', sa.Boolean(), default=False),
        sa.Column('has_faces', sa.Boolean(), default=False),
        sa.Column('visual_complexity', sa.Float(), nullable=True),
        
        # Metadata
        sa.Column('model_version', sa.String(), nullable=True),
        sa.Column('extraction_time_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('NOW()')),
        
        sa.PrimaryKeyConstraint('feature_id'),
        sa.ForeignKeyConstraint(['keyframe_id'], ['keyframes.keyframe_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.video_id'], ondelete='CASCADE')
    )
    
    # Add indices for performance
    op.create_index('idx_enhanced_features_keyframe', 'enhanced_features', ['keyframe_id'])
    op.create_index('idx_enhanced_features_video', 'enhanced_features', ['video_id'])
    op.create_index('idx_enhanced_features_scene_type', 'enhanced_features', ['scene_type'])
    op.create_index('idx_enhanced_features_objects', 'enhanced_features', ['objects_detected'], postgresql_using='gin')
    op.create_index('idx_enhanced_features_has_text', 'enhanced_features', ['has_text'])
    op.create_index('idx_enhanced_features_has_faces', 'enhanced_features', ['has_faces'])
    
    # Create query_enhancements table for tracking query enhancement performance
    op.create_table('query_enhancements',
        sa.Column('enhancement_id', sa.String(), nullable=False),
        sa.Column('original_query', sa.String(), nullable=False),
        sa.Column('enhanced_query', sa.String(), nullable=False),
        sa.Column('query_type', sa.String(), nullable=True),
        sa.Column('entities_detected', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('actions_detected', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('scene_context', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('enhancement_method', sa.String(), nullable=True),  # template, llm, hybrid
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('NOW()')),
        
        sa.PrimaryKeyConstraint('enhancement_id')
    )
    
    # Add indices for query enhancement tracking
    op.create_index('idx_query_enhancements_original', 'query_enhancements', ['original_query'])
    op.create_index('idx_query_enhancements_type', 'query_enhancements', ['query_type'])
    op.create_index('idx_query_enhancements_created', 'query_enhancements', ['created_at'])
    
    # Create search_analytics table for enhanced search performance tracking
    op.create_table('search_analytics',
        sa.Column('search_id', sa.String(), nullable=False),
        sa.Column('original_query', sa.String(), nullable=False),
        sa.Column('enhanced_query', sa.String(), nullable=True),
        sa.Column('search_type', sa.String(), nullable=False),  # regular, enhanced
        sa.Column('results_count', sa.Integer(), nullable=False),
        sa.Column('query_time_ms', sa.Float(), nullable=False),
        sa.Column('enhancement_time_ms', sa.Float(), nullable=True),
        sa.Column('top_score', sa.Float(), nullable=True),
        sa.Column('user_session', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('NOW()')),
        
        sa.PrimaryKeyConstraint('search_id')
    )
    
    # Add indices for search analytics
    op.create_index('idx_search_analytics_query', 'search_analytics', ['original_query'])
    op.create_index('idx_search_analytics_type', 'search_analytics', ['search_type'])
    op.create_index('idx_search_analytics_created', 'search_analytics', ['created_at'])
    op.create_index('idx_search_analytics_performance', 'search_analytics', ['query_time_ms', 'results_count'])

def downgrade():
    # Drop indices first
    op.drop_index('idx_search_analytics_performance')
    op.drop_index('idx_search_analytics_created')
    op.drop_index('idx_search_analytics_type')
    op.drop_index('idx_search_analytics_query')
    
    op.drop_index('idx_query_enhancements_created')
    op.drop_index('idx_query_enhancements_type')
    op.drop_index('idx_query_enhancements_original')
    
    op.drop_index('idx_enhanced_features_has_faces')
    op.drop_index('idx_enhanced_features_has_text')
    op.drop_index('idx_enhanced_features_objects')
    op.drop_index('idx_enhanced_features_scene_type')
    op.drop_index('idx_enhanced_features_video')
    op.drop_index('idx_enhanced_features_keyframe')
    
    # Drop tables
    op.drop_table('search_analytics')
    op.drop_table('query_enhancements')
    op.drop_table('enhanced_features')
    op.create_table('enhanced_features',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('keyframe_id', sa.String(), nullable=False),
        sa.Column('shot_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        
        # Core embeddings
        sa.Column('clip_embedding', sa.LargeBinary(), nullable=True),
        sa.Column('enhanced_embedding', sa.LargeBinary(), nullable=True),
        sa.Column('visual_features', sa.LargeBinary(), nullable=True),
        sa.Column('semantic_features', sa.LargeBinary(), nullable=True),
        
        # Feature data (JSON)
        sa.Column('detected_objects', postgresql.JSONB(), nullable=True),
        sa.Column('scene_features', postgresql.JSONB(), nullable=True),
        sa.Column('action_features', postgresql.JSONB(), nullable=True),
        sa.Column('text_features', postgresql.JSONB(), nullable=True),
        sa.Column('audio_features', postgresql.JSONB(), nullable=True),
        
        # Searchable columns for fast filtering
        sa.Column('objects_detected', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('scene_type', sa.String(50), nullable=True),
        sa.Column('indoor_outdoor', sa.String(20), nullable=True),
        sa.Column('lighting', sa.String(20), nullable=True),
        sa.Column('time_of_day', sa.String(20), nullable=True),
        sa.Column('has_text', sa.Boolean(), default=False),
        sa.Column('has_faces', sa.Boolean(), default=False),
        sa.Column('primary_action', sa.String(50), nullable=True),
        sa.Column('dominant_colors', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('composition_type', sa.String(20), nullable=True),
        
        # Searchable tags
        sa.Column('searchable_tags', postgresql.ARRAY(sa.String()), nullable=True),
        
        # Metadata
        sa.Column('extraction_time', sa.Float(), nullable=True),
        sa.Column('model_versions', postgresql.JSONB(), nullable=True),
        sa.Column('confidence_scores', postgresql.JSONB(), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        
        sa.ForeignKeyConstraint(['keyframe_id'], ['keyframes.keyframe_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['shot_id'], ['shots.shot_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.video_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indices for fast searching
    op.create_index('ix_enhanced_features_keyframe_id', 'enhanced_features', ['keyframe_id'])
    op.create_index('ix_enhanced_features_video_id', 'enhanced_features', ['video_id'])
    op.create_index('ix_enhanced_features_scene_type', 'enhanced_features', ['scene_type'])
    op.create_index('ix_enhanced_features_indoor_outdoor', 'enhanced_features', ['indoor_outdoor'])
    op.create_index('ix_enhanced_features_primary_action', 'enhanced_features', ['primary_action'])
    op.create_index('ix_enhanced_features_has_text', 'enhanced_features', ['has_text'])
    op.create_index('ix_enhanced_features_has_faces', 'enhanced_features', ['has_faces'])
    
    # Create GIN indices for array columns (fast array searching)
    op.create_index('ix_enhanced_features_objects_gin', 'enhanced_features', ['objects_detected'], 
                   postgresql_using='gin')
    op.create_index('ix_enhanced_features_tags_gin', 'enhanced_features', ['searchable_tags'], 
                   postgresql_using='gin')
    op.create_index('ix_enhanced_features_colors_gin', 'enhanced_features', ['dominant_colors'], 
                   postgresql_using='gin')
    
    # Create query_enhancements table for caching and learning
    op.create_table('query_enhancements',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('original_query', sa.String(), nullable=False),
        sa.Column('enhanced_query', sa.Text(), nullable=False),
        sa.Column('query_type', sa.String(50), nullable=True),
        sa.Column('entities', postgresql.JSONB(), nullable=True),
        sa.Column('actions', postgresql.JSONB(), nullable=True),
        sa.Column('scene_context', postgresql.JSONB(), nullable=True),
        sa.Column('temporal_context', postgresql.JSONB(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        
        # Performance tracking
        sa.Column('search_count', sa.Integer(), default=0),
        sa.Column('click_through_rate', sa.Float(), nullable=True),
        sa.Column('avg_relevance_score', sa.Float(), nullable=True),
        
        # Model info
        sa.Column('enhancement_method', sa.String(50), nullable=True),
        sa.Column('model_version', sa.String(50), nullable=True),
        
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indices for query enhancement table
    op.create_index('ix_query_enhancements_original', 'query_enhancements', ['original_query'])
    op.create_index('ix_query_enhancements_type', 'query_enhancements', ['query_type'])
    op.create_index('ix_query_enhancements_search_count', 'query_enhancements', ['search_count'])
    
    # Create search_analytics table for learning
    op.create_table('search_analytics',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=True),
        sa.Column('original_query', sa.String(), nullable=False),
        sa.Column('enhanced_query', sa.Text(), nullable=True),
        sa.Column('results_count', sa.Integer(), nullable=True),
        sa.Column('clicked_results', postgresql.JSONB(), nullable=True),
        sa.Column('user_feedback', postgresql.JSONB(), nullable=True),
        sa.Column('search_time_ms', sa.Float(), nullable=True),
        sa.Column('enhancement_time_ms', sa.Float(), nullable=True),
        
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create index for analytics
    op.create_index('ix_search_analytics_query', 'search_analytics', ['original_query'])
    op.create_index('ix_search_analytics_session', 'search_analytics', ['session_id'])
    op.create_index('ix_search_analytics_created_at', 'search_analytics', ['created_at'])

def downgrade():
    # Drop tables in reverse order
    op.drop_table('search_analytics')
    op.drop_table('query_enhancements')
    op.drop_table('enhanced_features')

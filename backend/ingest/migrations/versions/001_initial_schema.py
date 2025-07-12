"""Initial schema for video retrieval system

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create videos table
    op.create_table('videos',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=True),
        sa.Column('file_size', sa.BigInteger(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('fps', sa.Float(), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_videos_filename', 'videos', ['filename'])
    op.create_index('ix_videos_processed_at', 'videos', ['processed_at'])

    # Create shots table
    op.create_table('shots',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('shot_index', sa.Integer(), nullable=False),
        sa.Column('start_frame', sa.Integer(), nullable=False),
        sa.Column('end_frame', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('keyframe_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_shots_video_id', 'shots', ['video_id'])
    op.create_index('ix_shots_start_time', 'shots', ['start_time'])

    # Create keyframes table
    op.create_table('keyframes',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('shot_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('frame_index', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('thumbnail_path', sa.String(), nullable=True),
        sa.Column('features_extracted', sa.Boolean(), default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['shot_id'], ['shots.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_keyframes_video_id', 'keyframes', ['video_id'])
    op.create_index('ix_keyframes_timestamp', 'keyframes', ['timestamp'])

    # Create embeddings table
    op.create_table('embeddings',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('keyframe_id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('embedding_dim', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['keyframe_id'], ['keyframes.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_embeddings_video_id', 'embeddings', ['video_id'])
    op.create_index('ix_embeddings_model_name', 'embeddings', ['model_name'])

    # Create search_logs table for analytics
    op.create_table('search_logs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('results_count', sa.Integer(), nullable=False),
        sa.Column('query_time_ms', sa.Float(), nullable=False),
        sa.Column('user_ip', sa.String(), nullable=True),
        sa.Column('user_agent', sa.String(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_search_logs_timestamp', 'search_logs', ['timestamp'])
    op.create_index('ix_search_logs_query', 'search_logs', ['query'])

    # Create processing_jobs table for tracking async tasks
    op.create_table('processing_jobs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('video_id', sa.String(), nullable=False),
        sa.Column('job_type', sa.String(), nullable=False),  # 'shot_detection', 'keyframe_extraction', 'embedding_generation'
        sa.Column('status', sa.String(), nullable=False),    # 'pending', 'running', 'completed', 'failed'
        sa.Column('progress_percent', sa.Float(), default=0.0, nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_processing_jobs_video_id', 'processing_jobs', ['video_id'])
    op.create_index('ix_processing_jobs_status', 'processing_jobs', ['status'])

def downgrade():
    op.drop_table('processing_jobs')
    op.drop_table('search_logs')
    op.drop_table('embeddings')
    op.drop_table('keyframes')
    op.drop_table('shots')
    op.drop_table('videos')

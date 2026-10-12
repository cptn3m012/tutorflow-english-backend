"""Persist local lesson pictures and background jobs."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_lesson_pictures"
down_revision = "0001_saved_lessons"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "lesson_images",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("lesson_id", sa.String(36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata_json", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
    )
    op.create_index("ix_lesson_images_lesson_id", "lesson_images", ["lesson_id"])
    op.create_table(
        "picture_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("lesson_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("style", sa.String(16), nullable=False),
        sa.Column("model", sa.String(80), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("asset_id", sa.String(36), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    for name in ("lesson_id", "status", "created_at"):
        op.create_index(f"ix_picture_jobs_{name}", "picture_jobs", [name])
    active = sa.text("status IN ('queued', 'planning', 'rendering', 'checking')")
    op.create_index("uq_picture_job_active_lesson", "picture_jobs", ["lesson_id"], unique=True,
                    postgresql_where=active, sqlite_where=active)


def downgrade():
    op.drop_table("picture_jobs")
    op.drop_table("lesson_images")

"""Create the shared lesson library."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001_saved_lessons"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "saved_lessons",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("saved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("favorite", sa.Boolean(), nullable=False),
        sa.Column("level", sa.String(2), nullable=False),
        sa.Column("duration", sa.Integer(), nullable=False),
        sa.Column("theme", sa.String(500), nullable=False),
        sa.Column("mode", sa.String(4), nullable=False),
        sa.Column("lesson", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
    )
    op.create_index("ix_saved_lessons_saved_at", "saved_lessons", ["saved_at"])
    op.create_index("ix_saved_lessons_level", "saved_lessons", ["level"])


def downgrade():
    op.drop_table("saved_lessons")

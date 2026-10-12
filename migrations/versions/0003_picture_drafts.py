"""Keep picture working copies separate from the saved lesson library."""
from alembic import op
import sqlalchemy as sa

revision = "0003_picture_drafts"
down_revision = "0002_lesson_pictures"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("saved_lessons", sa.Column("is_draft", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade():
    op.drop_column("saved_lessons", "is_draft")

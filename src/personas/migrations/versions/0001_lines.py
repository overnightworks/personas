"""The lines every agent read, one row for each line of a window."""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.create_table(
        "lines",
        sa.Column("client_id", sa.Text, nullable=False),
        sa.Column("app", sa.Text, nullable=False),
        sa.Column("place_kind", sa.Text, nullable=False),
        sa.Column("place_id", sa.Text, nullable=False),
        sa.Column("generation", sa.Text, nullable=False),
        sa.Column("seq", sa.Numeric(scale=0), nullable=False),
        sa.Column("first_read_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("speaker", sa.Text, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("own", sa.Boolean, nullable=False),
        sa.Column("spoken", sa.Boolean, nullable=False),
        sa.PrimaryKeyConstraint("client_id", "app", "place_kind", "place_id", "generation", "seq"),
    )

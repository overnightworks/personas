"""A line the app marked as cut before its end."""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    # Earlier rows were read from windows that cannot carry the mark, so false is their true value; the default
    # only fills them, and a writer that forgets `cut` then fails instead of storing a silent false.
    op.add_column("lines", sa.Column("cut", sa.Boolean, nullable=False, server_default=sa.false()))
    op.alter_column("lines", "cut", server_default=None)

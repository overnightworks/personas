"""A line the app marked as an act: it says what someone did, not what someone said."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    # Earlier rows were read by a runner whose library refuses a window that carries the mark, so false is their true
    # value; the default only fills them, and a writer that forgets `act` then fails instead of storing a silent false.
    op.add_column("lines", sa.Column("act", sa.Boolean, nullable=False, server_default=sa.false()))
    op.alter_column("lines", "act", server_default=None)

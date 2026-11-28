"""The record: every entry an agent's history took, kept past the history's cut, with the room it was taken in."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"


def upgrade() -> None:
    # Expand only: a pod one revision behind never reads the table and cuts its history as before, so it serves while
    # the rollout runs. An entry is jsonb here, which no turn is told from, and its ordinal is null for an entry no turn
    # is told.
    op.create_table(
        "kept_entries",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("client_id", sa.Text, nullable=False),
        sa.Column("ordinal", sa.BigInteger),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("app", sa.Text, nullable=False),
        sa.Column("place_kind", sa.Text),
        sa.Column("place_id", sa.Text),
        sa.Column("generation", sa.Text),
        sa.Column("own", sa.Boolean, nullable=False),
        sa.Column("speaker", sa.Text),
        sa.Column("entry", postgresql.JSONB, nullable=False),
    )
    # A told entry's ordinal is its key, so writing it again changes nothing; rows without one are not constrained.
    op.create_index("kept_entries_client_ordinal", "kept_entries", ["client_id", "ordinal"], unique=True)

"""The mind's history in the database: each agent's entries and answered turns, and the generation of his owner."""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"


def upgrade() -> None:
    # An entry and a turn are JSON in text, not jsonb: jsonb orders an object's keys its own way, and a turn must be
    # told its history byte for byte as before.
    op.create_table(
        "history_entries",
        sa.Column("client_id", sa.Text, nullable=False),
        sa.Column("ordinal", sa.BigInteger, nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry", sa.Text, nullable=False),
        sa.PrimaryKeyConstraint("client_id", "ordinal"),
    )
    op.create_table(
        "answered_turns",
        sa.Column("client_id", sa.Text, nullable=False),
        sa.Column("ordinal", sa.BigInteger, nullable=False),
        sa.Column("through", sa.BigInteger, nullable=False),
        sa.Column("turn", sa.Text, nullable=False),
        sa.PrimaryKeyConstraint("client_id", "ordinal"),
    )
    op.create_table(
        "agent_owner",
        sa.Column("client_id", sa.Text, primary_key=True),
        sa.Column("generation", sa.BigInteger, nullable=False),
    )

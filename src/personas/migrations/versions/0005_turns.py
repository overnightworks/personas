"""Each agent's turns in the database: where reading his places stands, his due places and his pause, kept with his
history so that a pod that takes him over neither loses nor repeats a turn."""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"


def upgrade() -> None:
    # Expand only: a pod one revision behind never reads the table, and keeps serving while the rollout runs.
    op.create_table(
        "agent_turns",
        sa.Column("client_id", sa.Text, primary_key=True),
        sa.Column("turns", sa.Text, nullable=False),
        sa.Column("generation", sa.BigInteger, nullable=False),
    )

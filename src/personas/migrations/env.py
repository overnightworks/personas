from alembic import context

# ``personas.memory`` hands in the connection that holds the migration lock.
connection = context.config.attributes["connection"]
context.configure(connection=connection)
with context.begin_transaction():
    context.run_migrations()

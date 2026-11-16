"""A configuration directory for tests: the secret, key and persona files and a TOML that points at them."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from personas.config import CONFIG_ENVIRONMENT_VARIABLE
from tests.support.realm import API_GROUP

if TYPE_CHECKING:
    from pathlib import Path

    from personas.config import DatabaseTarget

PERSONA_NAME = "mozart"
GATEWAY_ORIGIN = "http://gateway.test"
MAX_TOOL_CALLS = 4
PERSONA_MAX_DURATION_SECONDS = 30


class Files:
    """A configuration directory, each file and the TOML changeable by a test.

    The valid text binds the ``say`` tool in its app, so it needs the gateway and a persona for its agent.
    """

    def __init__(self, directory: Path) -> None:
        self.secret = secrets.token_hex(16)
        self.secret_path = directory / "mozart.secret"
        self.secret_path.write_text(f"{self.secret}\n", encoding="utf-8")
        self.gateway_key = secrets.token_hex(16)
        self.key_path = directory / "gateway.key"
        self.key_path.write_text(f"{self.gateway_key}\n", encoding="utf-8")
        self.persona_prompt = secrets.token_hex(16)
        self.persona_path = directory / f"{PERSONA_NAME}.md"
        self.persona_path.write_text(self.persona_text(), encoding="utf-8")
        self.password = secrets.token_hex(16)
        self.password_path = directory / "memory.password"
        self.password_path.write_text(f"{self.password}\n", encoding="utf-8")
        self.ready_path = directory / "ready"
        self.config_path = directory / "personas.toml"
        self.text = self.valid_text()

    def persona_text(
        self, max_duration_seconds: int = PERSONA_MAX_DURATION_SECONDS, voice_window_bytes: int | None = None
    ) -> str:
        voice = "" if voice_window_bytes is None else f"voice: {{windowBytes: {voice_window_bytes}}}\n"
        return (
            f"---\nname: {PERSONA_NAME}\ndescription: persona-{secrets.token_hex(4)}\nappRoles: [admin, user]\n"
            f"model: text/{secrets.token_hex(4)}\nmaxDurationSeconds: {max_duration_seconds}\n{voice}---\n"
            f"{self.persona_prompt}\n"
        )

    def valid_text(self, mcp_url: str = "https://hopin.test/mcp", scope: str = "mcp-hopin", *, binds_tools=True) -> str:
        gateway = tools = persona = ""
        if binds_tools:
            gateway = (
                f'[gateway]\nbackend = "litellm"\napi_group = "{API_GROUP}"\n'
                f'[gateway.gateway]\norigin = "{GATEWAY_ORIGIN}"\nkey_file = "{self.key_path}"\n'
            )
            tools = f'tools = ["say"]\nmax_tool_calls = {MAX_TOOL_CALLS}\n'
            persona = f'persona_file = "{self.persona_path}"\n'
        return (
            'token_endpoint = "https://realm.test/protocol/openid-connect/token"\n'
            f'ready_file = "{self.ready_path}"\n{self.database_table()}{gateway}'
            f'[[apps]]\nmcp_url = "{mcp_url}"\nscope = "{scope}"\n{tools}'
            f'[[agents]]\nclient_id = "agent-mozart"\nsecret_file = "{self.secret_path}"\n{persona}'
        )

    @staticmethod
    def _table(name: str, fields: dict[str, str], overrides: dict[str, str | None]) -> str:
        """A table; an override replaces a field's TOML value, or drops the field when None."""
        for field, value in overrides.items():
            if value is None:
                del fields[field]
            else:
                fields[field] = value
        return f"[{name}]\n" + "".join(f"{field} = {value}\n" for field, value in fields.items())

    def database_table(self, **overrides: str | None) -> str:
        """A valid database table."""
        fields = {
            "host": '"db.test"',
            "port": "5432",
            "dbname": '"personas"',
            "user": '"mozart"',
            "password_file": f'"{self.password_path}"',
        }
        return self._table("database", fields, overrides)

    def set_database(self, **overrides: str | None) -> None:
        """Replaces the valid text's database table with one that has these overrides."""
        self.text = self.text.replace(self.database_table(), self.database_table(**overrides))

    def use_database(self, target: DatabaseTarget) -> None:
        """Points the valid text's database table at this database."""
        self.set_database(
            host=f'"{target.host}"', port=str(target.port), dbname=f'"{target.dbname}"', user=f'"{target.user}"'
        )

    def memory_table(self, **overrides: str | None) -> str:
        """A valid memory table: the defaults."""
        return self._table("memory", {}, overrides)

    def write(self) -> dict[str, str]:
        self.config_path.write_text(self.text, encoding="utf-8")
        return {CONFIG_ENVIRONMENT_VARIABLE: str(self.config_path)}

"""A configuration directory for tests: a secret file and a TOML that points at it."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from agents.config import CONFIG_ENVIRONMENT_VARIABLE

if TYPE_CHECKING:
    from pathlib import Path


class Files:
    """A configuration directory: a secret file and a TOML that points at it, each changeable by a test."""

    def __init__(self, directory: Path) -> None:
        self.secret = secrets.token_hex(16)
        self.secret_path = directory / "mozart.secret"
        self.secret_path.write_text(f"{self.secret}\n", encoding="utf-8")
        self.ready_path = directory / "ready"
        self.config_path = directory / "agents.toml"
        self.text = self.valid_text()

    def valid_text(self) -> str:
        return (
            'token_endpoint = "https://realm.test/protocol/openid-connect/token"\n'
            f'ready_file = "{self.ready_path}"\n'
            '[[apps]]\nmcp_url = "https://hopin.test/mcp"\nscope = "mcp-hopin"\n'
            f'[[agents]]\nclient_id = "agent-mozart"\nsecret_file = "{self.secret_path}"\n'
        )

    def write(self) -> dict[str, str]:
        self.config_path.write_text(self.text, encoding="utf-8")
        return {CONFIG_ENVIRONMENT_VARIABLE: str(self.config_path)}

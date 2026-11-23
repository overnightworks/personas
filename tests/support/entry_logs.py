"""The entry point's logging, as a test captures it."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from personas.main import configure_logging
from tests.support.logs import FormatterWithFields

if TYPE_CHECKING:
    import pytest


def configure_logging_into(caplog: pytest.LogCaptureFixture) -> None:
    """The entry point's logging, with the lines of every logger also in ``caplog`` and its fields in their text."""
    configure_logging()
    caplog.handler.setFormatter(FormatterWithFields())
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG)

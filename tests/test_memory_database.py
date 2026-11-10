"""The tests' database: a test that needs it fails, naming the wrapper that provides it, when none is given."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.postgres import URL_ENVIRONMENT_VARIABLE, WRAPPER, server_url


def test_a_test_without_a_database_fails_and_names_the_wrapper_that_provides_one(monkeypatch):
    monkeypatch.delenv(URL_ENVIRONMENT_VARIABLE, raising=False)

    with pytest.raises(pytest.fail.Exception) as failed:
        server_url()

    assert WRAPPER in str(failed.value)
    assert (Path(__file__).parent.parent / WRAPPER).is_file()

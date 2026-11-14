"""The configuration file, read through the entry point: a bad value stops the start and names itself."""

from __future__ import annotations

import math
import secrets
from typing import TYPE_CHECKING, NoReturn

import pytest

from personas.config import (
    CONFIG_ENVIRONMENT_VARIABLE,
    DEFAULT_CALL_SECONDS,
    DEFAULT_HISTORY_BYTES,
    DEFAULT_LINES_KEEP_DAYS,
    DEFAULT_SWEEP_MINUTES,
    DEFAULT_TODAY,
    DEFAULT_WRITE_SECONDS,
    ToolBinding,
    Voice,
    load_config,
)
from personas.decision import DEFAULT_BUDGET_PAUSE_CAP_SECONDS
from personas.main import CONFIGURATION_ERROR_EXIT_CODE, main
from personas.recall import TodayBounds
from tests.support.configfiles import MAX_TOOL_CALLS, PERSONA_NAME
from tests.support.realm import API_GROUP

if TYPE_CHECKING:
    from personas.config import Config
    from personas.readiness import ReadyMarker

APP_TABLE = (
    '[[apps]]\nmcp_url = "https://hopin.test/mcp"\nscope = "mcp-hopin"\n'
    f'tools = ["say"]\nmax_tool_calls = {MAX_TOOL_CALLS}\n'
)


TOOLS_ONLY_TABLE = (
    '[[apps]]\nmcp_url = "https://songmaker.test/mcp"\nscope = "mcp-songmaker"\nlistens = false\n'
    'tools = ["create_song", "generate_song"]\nmax_tool_calls = 5\n'
)


def _with_ready_file(text: str, value: str | None) -> str:
    """The valid text with its ready_file line removed (None) or set to a value."""
    others = [line for line in text.splitlines(keepends=True) if not line.startswith("ready_file")]
    return "".join(others if value is None else [f'ready_file = "{value}"\n', *others])


def never_composed(config: Config, marker: ReadyMarker) -> NoReturn:
    raise AssertionError("a refused configuration started the runner")


def fails_at_start(environment: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.delenv(CONFIG_ENVIRONMENT_VARIABLE, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as stopped:
        main(never_composed)
    assert stopped.value.code == CONFIGURATION_ERROR_EXIT_CODE
    return capsys.readouterr().err


def test_a_valid_file_gives_every_agent_its_secret_and_every_app_its_scope(files):
    config = load_config(files.write())

    assert config.token_endpoint.startswith("https://realm.test/")
    assert config.ready_file == files.ready_path
    assert [(app.mcp_url, app.scope) for app in config.apps] == [("https://hopin.test/mcp", "mcp-hopin")]
    assert [(agent.client_id, agent.client_secret.reveal()) for agent in config.agents] == [
        ("agent-mozart", files.secret)
    ]


def test_an_environment_without_the_variable_stops_the_start(monkeypatch, capsys):
    error = fails_at_start({}, monkeypatch, capsys)

    assert CONFIG_ENVIRONMENT_VARIABLE in error


def test_a_file_that_cannot_be_read_stops_the_start_and_names_its_path(files, monkeypatch, capsys):
    missing = files.config_path.parent / "nowhere.toml"

    error = fails_at_start({CONFIG_ENVIRONMENT_VARIABLE: str(missing)}, monkeypatch, capsys)

    assert str(missing) in error


@pytest.mark.parametrize(
    ("break_it", "named"),
    [
        (lambda text: text.replace("token_endpoint", "token_url"), "token_endpoint"),
        (lambda text: text.replace("https://realm.test/protocol/openid-connect/token", "realm"), "token_endpoint"),
        (lambda text: text.replace('mcp_url = "https://hopin.test/mcp"', 'mcp_url = "ftp://x"'), "apps.0.mcp_url"),
        (lambda text: text.replace('scope = "mcp-hopin"', 'scope = ""'), "apps.0.scope"),
        (lambda text: text.replace('scope = "mcp-hopin"\n', ""), "apps.0.scope"),
        (lambda text: text.replace('client_id = "agent-mozart"', 'client_id = ""'), "agents.0.client_id"),
        (lambda text: text.replace('client_id = "agent-mozart"\n', ""), "agents.0.client_id"),
        (lambda text: text.replace("secret_file", "secret"), "agents.0.secret_file"),
        (lambda text: text.replace(APP_TABLE, ""), "apps"),
        (lambda text: text.split("[[agents]]")[0], "agents"),
        (lambda text: "apps = []\n" + text.replace(APP_TABLE, ""), "apps"),
        (lambda text: "agents = []\n" + text.split("[[agents]]")[0], "agents"),
        (lambda text: text + "unknown = 1\n", "unknown"),
        (lambda text: text + "max_turns = 50\nturn_period_seconds = 3600\n", "max_turns"),
        (lambda text: text + "max_turns = 50\nturn_period_seconds = 3600\n", "turn_period_seconds"),
        (lambda text: text + APP_TABLE, "apps"),
        (lambda text: text + "[[agents]]" + text.split("[[agents]]")[1], "agents"),
        (lambda text: text.replace("[[agents]]", "[[agents"), "TOML"),
        (lambda text: _with_ready_file(text, None), "ready_file"),
        (lambda text: _with_ready_file(text, ""), "ready_file"),
        (lambda text: _with_ready_file(text, "ready"), "ready_file"),
    ],
    ids=[
        "token endpoint missing",
        "token endpoint not a URL",
        "app URL not http",
        "app scope empty",
        "app scope missing",
        "client id empty",
        "client id missing",
        "secret file missing",
        "no apps",
        "no agents",
        "empty list of apps",
        "empty list of agents",
        "unknown field",
        "max_turns set",
        "turn_period_seconds set",
        "app URL listed twice",
        "client id listed twice",
        "not TOML",
        "ready file missing",
        "ready file empty",
        "ready file relative",
    ],
)
def test_a_missing_or_malformed_field_stops_the_start_and_names_it(files, monkeypatch, capsys, break_it, named):
    files.text = break_it(files.text)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert named in error
    assert files.secret not in error


@pytest.mark.parametrize(
    "arrange",
    [
        lambda files: files.secret_path.unlink(),
        lambda files: files.secret_path.write_text("  \n", encoding="utf-8"),
        lambda files: files.secret_path.write_bytes(b"\xff\xfe"),
        lambda files: setattr(files, "text", files.text.replace(str(files.secret_path), "mozart.secret")),
    ],
    ids=["missing", "empty", "not text", "relative path"],
)
def test_a_secret_file_that_is_unusable_stops_the_start_and_names_the_field(files, monkeypatch, capsys, arrange):
    arrange(files)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert "agents.0.secret_file" in error


def test_a_secret_written_into_the_file_is_refused_without_being_echoed(files, monkeypatch, capsys):
    files.text += f'client_secret = "{files.secret}"\n'

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert "client_secret" in error
    assert files.secret not in error


def test_a_valid_file_gives_the_agent_its_persona_the_app_its_tools_and_the_runner_its_gateway(files):
    config = load_config(files.write())

    [agent], [app] = config.agents, config.apps
    assert agent.persona is not None
    assert (agent.persona.name, agent.persona.prompt) == (PERSONA_NAME, files.persona_prompt)
    assert app.binding == ToolBinding(("say",), MAX_TOOL_CALLS)
    assert config.gateway is not None
    assert (config.gateway.api_group, config.gateway.gateway.key_file) == (API_GROUP, files.key_path)


def figure_table(url: str | None = None, sha256: str | None = None, extra: str = "") -> str:
    url = f"https://figures.test/{secrets.token_hex(6)}.glb" if url is None else url
    sha256 = secrets.token_hex(32) if sha256 is None else sha256
    return f'[agents.figure]\nurl = "{url}"\nsha256 = "{sha256}"\n{extra}'


def test_an_agent_entry_may_name_the_figure_he_brings_and_none_otherwise(files):
    assert load_config(files.write()).agents[0].figure is None
    url, digest = f"https://figures.test/{secrets.token_hex(6)}.glb", secrets.token_hex(32)
    files.text += figure_table(url, digest)

    [agent] = load_config(files.write()).agents

    assert agent.figure is not None
    assert (agent.figure.url, agent.figure.sha256) == (url, digest)
    assert url not in repr(agent)
    assert digest not in repr(agent)


@pytest.mark.parametrize(
    ("table", "named"),
    [
        (figure_table(url="http://figures.test/figure.glb"), "agents.0.figure.url"),
        (figure_table(url="https://figures.test/figure.glb?size=big"), "agents.0.figure.url"),
        (figure_table(url="/figure.glb"), "agents.0.figure.url"),
        (figure_table(sha256="ab" * 31), "agents.0.figure.sha256"),
        (figure_table(sha256="AB" * 32), "agents.0.figure.sha256"),
        (figure_table(extra='size = "big"\n'), "agents.0.figure.size"),
        ('[agents.figure]\nurl = "https://figures.test/figure.glb"\n', "agents.0.figure.sha256"),
    ],
    ids=["not https", "query", "not absolute", "digest short", "digest uppercase", "unknown key", "digest missing"],
)
def test_a_malformed_figure_stops_the_start_and_names_the_field(files, monkeypatch, capsys, table, named):
    files.text += table

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert named in error
    assert "figures.test" not in error


def test_an_agent_entry_turns_his_voice_on_with_a_table_and_may_name_it(files):
    assert load_config(files.write()).agents[0].voice is None
    files.text += "[agents.voice]\n"
    assert load_config(files.write()).agents[0].voice == Voice(None)
    files.text += 'name = "Rachel_V2.en-GB"\n'

    assert load_config(files.write()).agents[0].voice == Voice("Rachel_V2.en-GB")


@pytest.mark.parametrize(
    ("table", "named"),
    [
        ('[agents.voice]\nname = ""\n', "agents.0.voice.name"),
        ('[agents.voice]\nname = "two words"\n', "agents.0.voice.name"),
        ('[agents.voice]\nname = "a/b"\n', "agents.0.voice.name"),
        (f'[agents.voice]\nname = "{"a" * 65}"\n', "agents.0.voice.name"),
        ("[agents.voice]\nname = 7\n", "agents.0.voice.name"),
        ('[agents.voice]\nspeed = "fast"\n', "agents.0.voice.speed"),
        ('voice = "ryan"\n', "agents.0.voice"),
    ],
    ids=["empty", "space", "slash", "too long", "not a string", "unknown key", "not a table"],
)
def test_a_malformed_voice_stops_the_start_and_names_the_field_but_not_its_value(
    files, monkeypatch, capsys, table, named
):
    files.text += table

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert named in error
    for value in ("two words", "a/b", "ryan", "fast"):
        assert value not in error


@pytest.mark.parametrize("table", ["", "[agents.voice]\n"], ids=["without the table", "with the table"])
def test_a_persona_file_with_a_voice_section_stops_the_start_and_names_the_field_but_not_its_content(
    files, monkeypatch, capsys, table
):
    files.persona_path.write_text(files.persona_text(voice_window_bytes=8192), encoding="utf-8")
    files.text += table

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert "agents.0.persona_file" in error
    assert "voice" in error
    assert "8192" not in error
    assert files.persona_prompt not in error


def test_a_configuration_without_tools_in_any_app_needs_no_gateway_and_no_persona(files):
    files.text = files.valid_text(binds_tools=False)

    config = load_config(files.write())

    assert config.gateway is None
    assert [app.binding for app in config.apps] == [None]
    assert [agent.persona for agent in config.agents] == [None]


@pytest.mark.parametrize(
    ("break_it", "named"),
    [
        (lambda text: text.replace(f"max_tool_calls = {MAX_TOOL_CALLS}\n", ""), "apps.0"),
        (lambda text: text.replace('tools = ["say"]\n', ""), "apps.0"),
        (
            lambda text: text.replace(f"max_tool_calls = {MAX_TOOL_CALLS}", "max_tool_calls = 0"),
            "apps.0.max_tool_calls",
        ),
        (lambda text: text.replace('tools = ["say"]', 'tools = ["say", "say"]'), "apps.0"),
        (lambda text: text.replace('tools = ["say"]', 'tools = [""]'), "apps.0"),
        (lambda text: text.replace('tools = ["say"]', 'tools = ["say", "speak"]'), "apps.0.tools"),
        (lambda text: text.replace('[gateway]\nbackend = "litellm"\n', '[gatewa]\nbackend = "litellm"\n'), "gatewa"),
        (lambda text: text.split("[gateway]")[0] + "[[apps]]" + text.split("[[apps]]")[1], "gateway"),
        (lambda text: text.replace('backend = "litellm"', 'backend = "other"'), "gateway.backend"),
        (lambda text: text.replace("origin = ", "origins = "), "gateway.gateway"),
        (lambda text: text.replace("persona_file", "persona"), "persona"),
        (lambda text: "\n".join(line for line in text.splitlines() if "persona_file" not in line), "persona_file"),
    ],
    ids=[
        "tools without a call bound",
        "a call bound without tools",
        "no call allowed",
        "a tool listed twice",
        "an empty tool name",
        "speak among the tools",
        "unknown gateway key",
        "tools without a gateway",
        "unknown backend",
        "gateway origin missing",
        "persona file misnamed",
        "tools without a persona file",
    ],
)
def test_tools_need_their_call_bound_and_the_gateway_and_a_persona_to_run_with(
    files, monkeypatch, capsys, break_it, named
):
    files.text = break_it(files.text)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert named in error
    assert files.secret not in error
    assert files.gateway_key not in error


@pytest.mark.parametrize(
    "arrange",
    [
        lambda files: files.persona_path.unlink(),
        lambda files: files.persona_path.write_text("no front matter\n", encoding="utf-8"),
        lambda files: files.persona_path.write_text(files.persona_text().replace("name: mozart", "name: bach")),
        lambda files: files.persona_path.write_text(
            files.persona_text().replace("---\n" + files.persona_prompt, "---\n")
        ),
        lambda files: files.persona_path.write_text(files.persona_text().replace("[admin, user]", "[user]")),
        lambda files: files.persona_path.write_text(files.persona_text().replace("[admin, user]", "[admin]")),
        lambda files: files.persona_path.write_text(
            files.persona_text()
            .replace("model: text/", "model: image/")
            .replace("---\n" + files.persona_prompt, "picture:\n  size: 64x64\n---\n" + files.persona_prompt)
        ),
        lambda files: files.persona_path.write_text(
            files.persona_text().replace("---\n" + files.persona_prompt, "tools: [say]\n---\n")
        ),
        lambda files: files.persona_path.write_text(
            files.persona_text().replace("maxDurationSeconds: 30", "maxDurationSeconds: 30\nmaxToolCalls: 3")
        ),
        lambda files: setattr(files, "text", files.text.replace(str(files.persona_path), "mozart.md")),
    ],
    ids=[
        "missing",
        "no front matter",
        "name not the file's",
        "no prompt",
        "roles invalid",
        "no user role",
        "picture agent",
        "undeclared tool",
        "call bound without tools",
        "relative path",
    ],
)
def test_a_persona_file_that_is_unusable_stops_the_start_and_names_the_field(files, monkeypatch, capsys, arrange):
    arrange(files)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert "agents.0.persona_file" in error
    assert files.persona_prompt not in error


def test_a_tools_only_app_has_its_tools_its_call_bound_and_the_default_call_bound_unless_set(files):
    files.text += TOOLS_ONLY_TABLE
    files.text += (
        TOOLS_ONLY_TABLE.replace("songmaker", "other").replace("create_song", "make").replace("generate_song", "play")
        + "call_seconds = 4.5\n"
    )

    hopin, songmaker, other = load_config(files.write()).apps

    assert (hopin.listens, songmaker.listens, other.listens) == (True, False, False)
    assert songmaker.binding == ToolBinding(("create_song", "generate_song"), 5)
    assert (songmaker.call_seconds, other.call_seconds) == (DEFAULT_CALL_SECONDS, 4.5)


@pytest.mark.parametrize(
    ("break_it", "named", "reason"),
    [
        (
            lambda text: text + TOOLS_ONLY_TABLE.replace("create_song", "say"),
            "apps",
            "a tool name is bound by two apps",
        ),
        (
            lambda text: text + TOOLS_ONLY_TABLE.replace("generate_song", "say").replace("songmaker", "x"),
            "apps",
            "a tool name is bound by two apps",
        ),
        (
            lambda text: (
                text
                + TOOLS_ONLY_TABLE
                + TOOLS_ONLY_TABLE.replace("songmaker", "other").replace('"create_song"', '"play"')
            ),
            "apps",
            "a tool name is bound by two apps",
        ),
        (
            lambda text: text + TOOLS_ONLY_TABLE.replace('tools = ["create_song", "generate_song"]\n', ""),
            "apps.1",
            "listens = false needs tools",
        ),
        (
            lambda text: text.replace('scope = "mcp-hopin"\n', 'scope = "mcp-hopin"\nlistens = false\n'),
            "apps",
            "no app listens",
        ),
        (lambda text: text + TOOLS_ONLY_TABLE + "call_seconds = 0\n", "apps.1.call_seconds", "greater than 0"),
        (lambda text: text + APP_TABLE.replace("hopin", "other"), "apps", "a tool name is bound by two apps"),
    ],
    ids=[
        "a tool name the first app has too",
        "a tool name a listening app has too",
        "a tool name another tools-only app has too",
        "a tools-only app without tools",
        "no listening app",
        "no call time",
        "a tool name another listening app has too",
    ],
)
def test_a_tool_name_two_apps_of_one_turn_share_and_a_file_without_a_listening_app_are_refused(
    files, monkeypatch, capsys, break_it, named, reason
):
    files.text = break_it(files.text)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert named in error
    assert reason in error


def test_a_listening_app_may_set_the_call_time_a_turn_of_another_app_has_for_each_request_to_it(files):
    files.text += APP_TABLE.replace("hopin", "other").replace('tools = ["say"]', 'tools = ["play"]') + (
        "call_seconds = 4.5\n"
    )

    hopin, other = load_config(files.write()).apps

    assert (hopin.call_seconds, other.call_seconds) == (DEFAULT_CALL_SECONDS, 4.5)


def test_a_file_that_binds_many_differently_named_tools_in_a_second_listening_app_starts(files):
    many = ", ".join(f'"{secrets.token_hex(4)}"' for _ in range(29))
    files.text += APP_TABLE.replace("hopin", "table").replace('tools = ["say"]', f"tools = [{many}]")

    hopin, table = load_config(files.write()).apps

    assert (len(hopin.binding.tools), len(table.binding.tools)) == (1, 29)


def test_a_memory_table_gives_the_runner_its_database_its_password_and_a_default_write_bound(files):
    files.text += files.memory_table()

    memory = load_config(files.write()).memory

    assert memory is not None
    assert (memory.host, memory.port, memory.dbname, memory.user) == ("db.test", 5432, "personas", "mozart")
    assert memory.password.reveal() == files.password
    assert memory.write_seconds == DEFAULT_WRITE_SECONDS


@pytest.mark.parametrize(("written", "bound"), [("0.5", 0.5), ("inf", math.inf)], ids=["seconds", "no bound"])
def test_a_memory_table_may_set_the_write_bound_or_leave_a_write_unbounded(files, written, bound):
    files.text += files.memory_table(write_seconds=written)

    memory = load_config(files.write()).memory

    assert memory is not None
    assert memory.write_seconds == bound


def test_a_memory_table_tells_a_turn_the_default_day_unless_it_sets_how_far_back_and_how_much(files):
    files.text += files.memory_table()
    default = load_config(files.write()).memory
    files.text = files.valid_text() + files.memory_table(today_hours="inf", today_bytes="1024")

    memory = load_config(files.write()).memory

    assert default is not None
    assert default.today == DEFAULT_TODAY
    assert memory is not None
    assert memory.today == TodayBounds(hours=math.inf, size_bytes=1024)


@pytest.mark.parametrize(
    ("written", "days", "minutes"),
    [
        ({}, DEFAULT_LINES_KEEP_DAYS, DEFAULT_SWEEP_MINUTES),
        ({"lines_keep_days": "inf", "sweep_minutes": "inf"}, math.inf, math.inf),
    ],
    ids=["defaults", "no limit"],
)
def test_a_memory_table_keeps_and_sweeps_by_default_unless_it_sets_either_or_leaves_it_infinite(
    files, written, days, minutes
):
    files.text += files.memory_table(**written)

    memory = load_config(files.write()).memory

    assert memory is not None
    assert memory.lines_keep_days == days
    assert memory.sweep_minutes == minutes


@pytest.mark.parametrize(
    ("written", "bytes_kept"),
    [(None, DEFAULT_HISTORY_BYTES), ("1024", 1024)],
    ids=["absent", "set"],
)
def test_the_history_is_bounded_by_the_default_unless_the_file_sets_history_bytes(files, written, bytes_kept):
    if written is not None:
        files.text = f"history_bytes = {written}\n{files.text}"

    assert load_config(files.write()).history_bytes == bytes_kept


@pytest.mark.parametrize("written", ["0", "-1", "1.5", '"1024"', "true"])
def test_a_history_bound_that_is_no_positive_integer_stops_the_start_and_names_the_field(
    files, monkeypatch, capsys, written
):
    files.text = f"history_bytes = {written}\n{files.text}"

    assert "history_bytes" in fails_at_start(files.write(), monkeypatch, capsys)


@pytest.mark.parametrize(
    ("written", "seconds"),
    [(None, DEFAULT_BUDGET_PAUSE_CAP_SECONDS), ("60.5", 60.5), ("1", 1.0)],
    ids=["absent", "set", "at the backoff base"],
)
def test_the_pause_for_a_spent_budget_is_capped_by_the_default_unless_the_file_sets_its_cap(files, written, seconds):
    if written is not None:
        files.text = f"budget_pause_cap_seconds = {written}\n{files.text}"

    assert load_config(files.write()).budget_pause_cap_seconds == seconds


@pytest.mark.parametrize("written", ["0", "0.5", "-1", '"60"', "true", "inf", "nan"])
def test_a_pause_cap_below_the_backoff_base_or_no_number_stops_the_start_and_names_the_field(
    files, monkeypatch, capsys, written
):
    files.text = f"budget_pause_cap_seconds = {written}\n{files.text}"

    assert "budget_pause_cap_seconds" in fails_at_start(files.write(), monkeypatch, capsys)


def test_a_file_without_a_memory_table_opens_no_store(files):
    assert load_config(files.write()).memory is None


@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        ({"host": None}, "memory.host"),
        ({"host": '""'}, "memory.host"),
        ({"port": None}, "memory.port"),
        ({"port": '"5432"'}, "memory.port"),
        ({"port": "70000"}, "memory.port"),
        ({"dbname": None}, "memory.dbname"),
        ({"user": None}, "memory.user"),
        ({"password_file": None}, "memory.password_file"),
        ({"write_seconds": "0"}, "memory.write_seconds"),
        ({"write_seconds": '"2"'}, "memory.write_seconds"),
        ({"write_seconds": "nan"}, "memory.write_seconds"),
        ({"today_hours": "0"}, "memory.today_hours"),
        ({"today_hours": '"24"'}, "memory.today_hours"),
        ({"today_hours": "nan"}, "memory.today_hours"),
        ({"today_bytes": "0"}, "memory.today_bytes"),
        ({"lines_keep_days": "0"}, "memory.lines_keep_days"),
        ({"lines_keep_days": "-3"}, "memory.lines_keep_days"),
        ({"lines_keep_days": '"30"'}, "memory.lines_keep_days"),
        ({"lines_keep_days": "nan"}, "memory.lines_keep_days"),
        ({"sweep_minutes": "0"}, "memory.sweep_minutes"),
        ({"sweep_minutes": "-3"}, "memory.sweep_minutes"),
        ({"sweep_minutes": '"60"'}, "memory.sweep_minutes"),
        ({"sweep_minutes": "nan"}, "memory.sweep_minutes"),
        ({"today_bytes": "1.5"}, "memory.today_bytes"),
        ({"password": '"in-the-file"'}, "memory.password"),
    ],
    ids=[
        "host missing",
        "host empty",
        "port missing",
        "port a string",
        "port out of range",
        "database missing",
        "user missing",
        "password file missing",
        "write bound zero",
        "write bound a string",
        "write bound no number",
        "day of no hours",
        "day hours a string",
        "day hours no number",
        "day of no bytes",
        "keep period of no days",
        "keep period negative",
        "keep period a string",
        "keep period no number",
        "sweep period of no minutes",
        "sweep period negative",
        "sweep period a string",
        "sweep period no number",
        "day bytes a fraction",
        "password written into the file",
    ],
)
def test_a_missing_or_malformed_memory_field_stops_the_start_and_names_it(files, monkeypatch, capsys, overrides, named):
    files.text += files.memory_table(**overrides)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert named in error
    assert "in-the-file" not in error


@pytest.mark.parametrize(
    "arrange",
    [
        lambda files: files.password_path.unlink(),
        lambda files: files.password_path.write_text("  \n", encoding="utf-8"),
        lambda files: setattr(files, "text", files.text.replace(str(files.password_path), "memory.password")),
    ],
    ids=["missing", "empty", "relative path"],
)
def test_a_password_file_that_is_unusable_stops_the_start_and_names_the_field(files, monkeypatch, capsys, arrange):
    files.text += files.memory_table()
    arrange(files)

    error = fails_at_start(files.write(), monkeypatch, capsys)

    assert "memory.password_file" in error

from pathlib import Path

import pytest
from agents.agent_file import load_agent

CHARACTERS = Path(__file__).resolve().parents[1] / "src" / "characters"
TABLE_PARAGRAPH_START = "Am Spieltisch:"
TALKED_OVER_PARAGRAPH_START = "Wenn man über dich hinwegspricht:"
CONCEDING_RULE = (
    "nicht mehr gewinnen kannst, gibst du auf, wenn die App das anbietet",
    "ein verlorenes Spiel aufzugeben, stimmst du zu",
)
APP_AND_TOOL_NAMES = ("Skat", "give_up_game", "agree_to_give_up", "decline_give_up")


def paragraphs_starting(persona: Path, start: str) -> list[str]:
    return [line for line in persona.read_text(encoding="utf-8").splitlines() if line.startswith(start)]


def table_paragraphs(persona: Path) -> list[str]:
    return paragraphs_starting(persona, TABLE_PARAGRAPH_START)


@pytest.mark.parametrize("persona", sorted(CHARACTERS.glob("*.md")), ids=lambda path: path.stem)
def test_every_persona_file_says_in_its_table_paragraph_when_to_concede(persona):
    paragraphs = table_paragraphs(persona)

    assert len(paragraphs) == 1
    assert all(fragment in paragraphs[0] for fragment in CONCEDING_RULE)


@pytest.mark.parametrize("persona", sorted(CHARACTERS.glob("*.md")), ids=lambda path: path.stem)
def test_no_persona_file_names_an_app_or_tool_in_its_table_paragraph(persona):
    named = [name for paragraph in table_paragraphs(persona) for name in APP_AND_TOOL_NAMES if name in paragraph]

    assert named == []


@pytest.mark.parametrize("persona", sorted(CHARACTERS.glob("*.md")), ids=lambda path: path.stem)
def test_every_persona_file_has_one_talked_over_paragraph_that_reaches_the_turns_instructions(persona):
    paragraphs = paragraphs_starting(persona, TALKED_OVER_PARAGRAPH_START)

    assert len(paragraphs) == 1
    assert paragraphs[0] in load_agent(persona, []).prompt

from pathlib import Path

import pytest

CHARACTERS = Path(__file__).resolve().parents[1] / "src" / "characters"
TABLE_PARAGRAPH_START = "Am Spieltisch:"
CONCEDING_RULE = (
    "nicht mehr gewinnen kannst, gibst du auf, wenn die App das anbietet",
    "ein verlorenes Spiel aufzugeben, stimmst du zu",
)


@pytest.mark.parametrize("persona", sorted(CHARACTERS.glob("*.md")), ids=lambda path: path.stem)
def test_every_persona_file_says_in_its_table_paragraph_when_to_concede(persona):
    paragraphs = [
        line for line in persona.read_text(encoding="utf-8").splitlines() if line.startswith(TABLE_PARAGRAPH_START)
    ]

    assert len(paragraphs) == 1
    assert all(fragment in paragraphs[0] for fragment in CONCEDING_RULE)

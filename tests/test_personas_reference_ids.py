import ast
import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
DOCS = REPOSITORY / "docs"
REFERENCE_DOCUMENTS = ("turns.md", "configuration.md", "log.md")
DOCUMENTED_ID = re.compile(r"\(([A-Z]+-\d+)\)")
BULLET = re.compile(r"^- .*(?:\n  .*)*", re.MULTILINE)
TABLE_SEPARATOR = re.compile(r"\|[-| :]+\|")
CITATION = re.compile(r"[A-Z]+-\d+(?:, [A-Z]+-\d+)*(?=[:.])")


def documented_ids(docs: Path) -> set[str]:
    return {
        rule_id
        for name in REFERENCE_DOCUMENTS
        for rule_id in DOCUMENTED_ID.findall((docs / name).read_text(encoding="utf-8"))
    }


def rules_of(document: str) -> list[str]:
    """The document's bullets, each with its indented continuation, and the rows of its tables below their header."""
    rules = BULLET.findall(document)
    in_table_body = False
    for line in document.splitlines():
        if not line.startswith("|"):
            in_table_body = False
        elif TABLE_SEPARATOR.fullmatch(line):
            in_table_body = True
        elif in_table_body:
            rules.append(line)
    return rules


def rules_without_an_id(document: str) -> list[str]:
    return [rule for rule in rules_of(document) if not DOCUMENTED_ID.search(rule)]


def cited_ids() -> dict[str, set[str]]:
    """Each cited ID with the tests that cite it at the start of their docstring."""
    citations: dict[str, set[str]] = {}
    for module in (REPOSITORY / "tests").rglob("test_*.py"):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or not node.name.startswith("test_"):
                continue
            cited = CITATION.match(ast.get_docstring(node) or "")
            for rule_id in cited.group().split(", ") if cited else ():
                citations.setdefault(rule_id, set()).add(f"{module.relative_to(REPOSITORY)}::{node.name}")
    return citations


def test_every_id_a_test_cites_names_a_rule_of_a_reference_document():
    documented = documented_ids(DOCS)

    unknown = {rule_id: tests for rule_id, tests in cited_ids().items() if rule_id not in documented}

    assert unknown == {}


def test_every_rule_of_a_reference_document_is_cited_by_a_test():
    uncited = documented_ids(DOCS) - set(cited_ids())

    assert uncited == set()


@pytest.mark.parametrize("name", REFERENCE_DOCUMENTS)
def test_every_bullet_and_table_row_of_a_reference_document_carries_an_id(name):
    assert rules_without_an_id((DOCS / name).read_text(encoding="utf-8")) == []


def test_an_id_written_outside_the_reference_documents_names_no_rule(tmp_path):
    for name in REFERENCE_DOCUMENTS:
        (tmp_path / name).write_text("# A reference\n", encoding="utf-8")
    (tmp_path / "ARCHITECTURE.md").write_text(
        "See the window rule (TURNS-3) in the turns reference.\n", encoding="utf-8"
    )

    assert documented_ids(tmp_path) == set()


def test_a_bullet_or_a_table_row_without_an_id_is_a_rule_without_one_and_prose_or_a_header_is_none():
    document = (
        "Prose that is context.\n\n"
        "- A rule that carries its ID\n  on two lines. (TURNS-1)\n"
        "- A rule without one\n  on two lines.\n\n"
        "| Name | Meaning |\n|---|---|\n| `a` | a field with one (CONFIG-2) |\n| `b` | a field without one |\n"
    )

    assert rules_without_an_id(document) == ["- A rule without one\n  on two lines.", "| `b` | a field without one |"]

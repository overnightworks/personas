import ast
import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DOCUMENTED_ID = re.compile(r"\(([A-Z]+-\d+)\)")
CITATION = re.compile(r"[A-Z]+-\d+(?:, [A-Z]+-\d+)*(?=[:.])")

# Rules the reference documents state that no test proves yet; each leaves this set when a test cites it
# or its rule leaves the document.
UNPROVEN = frozenset({"TURNS-2", "TURNS-8", "CONFIG-25", "CONFIG-38", "LOG-32"})


def documented_ids() -> set[str]:
    return {
        rule_id
        for document in (REPOSITORY / "docs").rglob("*.md")
        for rule_id in DOCUMENTED_ID.findall(document.read_text(encoding="utf-8"))
    }


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
    documented = documented_ids()

    unknown = {rule_id: tests for rule_id, tests in cited_ids().items() if rule_id not in documented}

    assert unknown == {}


def test_every_rule_of_a_reference_document_is_cited_by_a_test_unless_it_is_named_unproven():
    uncited = documented_ids() - set(cited_ids())

    assert uncited == UNPROVEN

import ast
from pathlib import Path


CHECKED_FILES = (
    Path("autotrade/core/autonomous_agent.py"),
    Path("autotrade/core/day_manager.py"),
)


def _duplicate_methods(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    duplicates: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        seen: dict[str, int] = {}
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name in seen:
                duplicates.append(
                    f"{path}:{node.name}.{child.name} lines {seen[child.name]},{child.lineno}"
                )
            seen[child.name] = child.lineno
    return duplicates


def test_no_duplicate_method_names_inside_classes():
    duplicates = []
    for path in CHECKED_FILES:
        duplicates.extend(_duplicate_methods(path))

    assert duplicates == []

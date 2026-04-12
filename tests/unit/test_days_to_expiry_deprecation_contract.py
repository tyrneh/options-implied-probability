"""Architecture guardrails for days-to-expiry deprecation."""

from __future__ import annotations

import ast
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = [
    REPO_ROOT / "oipd" / "core",
    REPO_ROOT / "oipd" / "interface",
    REPO_ROOT / "oipd" / "pipelines",
]
REMOVED_DAYS_IDENTIFIER_PATTERN = re.compile(r"(?<!calendar_)days_to_expiry(?!\w)")


def test_active_code_paths_do_not_call_engine_helpers_with_days_to_expiry():
    """Active code should not route pricing/probability helper calls through day counts."""
    offenders: list[str] = []
    for root in SEARCH_ROOTS:
        for path in root.rglob("*.py"):
            module = ast.parse(path.read_text(), filename=str(path))
            if any(
                keyword.arg == "days_to_expiry"
                for node in ast.walk(module)
                if isinstance(node, ast.Call)
                for keyword in node.keywords
            ):
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_no_function_signatures_accept_days_to_expiry_arguments():
    """No runtime signatures should expose the removed day-count argument."""
    offenders: list[str] = []
    for root in SEARCH_ROOTS:
        for path in root.rglob("*.py"):
            module = ast.parse(path.read_text(), filename=str(path))
            accepts_days = False
            for node in ast.walk(module):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                positional_args = [arg.arg for arg in node.args.args]
                kwonly_args = [arg.arg for arg in node.args.kwonlyargs]
                if (
                    "days_to_expiry" in positional_args
                    or "days_to_expiry" in kwonly_args
                ):
                    accepts_days = True
                    break

            if accepts_days:
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_runtime_code_does_not_reference_removed_days_identifier():
    """Runtime code should not reference the removed identifier text directly."""
    offenders: list[str] = []
    for root in SEARCH_ROOTS:
        for path in root.rglob("*.py"):
            text = path.read_text()
            if REMOVED_DAYS_IDENTIFIER_PATTERN.search(text):
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []

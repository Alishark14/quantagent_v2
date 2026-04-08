"""Restricted Python execution sandbox for the Quant Data Scientist.

ARCHITECTURE.md §13.1.6 forbids the LLM from writing to anything but
``alpha_factors.json``. The simplest enforcement: never give the LLM
file-write capability in the first place. The agent runs LLM-generated
analysis code through this sandbox, which:

* Strips ``__builtins__`` down to a small whitelist (no ``open``, no
  ``__import__``, no ``exec``, no ``eval``, no ``compile``).
* Pre-binds the data science libraries the analysis script is allowed
  to use (``pandas``, ``numpy``, ``scipy.stats``,
  ``statsmodels.stats.multitest.multipletests``).
* Pre-binds the trade + OHLCV inputs as ``trades_df`` and ``ohlcv``.
* Reads back the ``result`` variable and returns it.

This is NOT a security boundary against an adversarial LLM — Python's
exec sandbox is famously porous. It IS a defence-in-depth layer that
catches most accidental misbehaviour: the LLM forgetting it can't open
files, the LLM writing a print loop, the LLM trying to import the
filesystem. Combined with code review of LangSmith traces it's
sufficient for an offline cron worker.

For a real adversarial threat model the agent would need a separate
process / container with seccomp + cgroups. That's a Phase 4 lift —
documented as a deviation in CHANGELOG.md.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# A *small* whitelist of safe builtins. Anything not on this list is
# unavailable inside the sandbox. The list is intentionally conservative
# — it's much easier to whitelist a missing builtin later than to deny
# a dangerous one we forgot about.
_SAFE_BUILTINS: dict[str, Any] = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "range",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        # The Python data-science idiom needs these too.
        "True",
        "False",
        "None",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "Exception",
    )
}


# Names that, if they appear as substrings in the code, are an
# instant rejection. Belt-and-braces over the builtin whitelist —
# guards against creative escapes like getattr(0, "__class__").
_FORBIDDEN_PATTERNS = (
    "__import__",
    "__builtins__",
    "__class__",
    "__bases__",
    "__subclasses__",
    "__globals__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "globals(",
    "locals(",
    "input(",
    "breakpoint(",
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import shutil",
    "import pathlib",
    "from os",
    "from sys",
    "from subprocess",
    "from socket",
    "from shutil",
    "from pathlib",
)


class SandboxRejected(Exception):
    """Raised when the LLM-generated code contains a forbidden pattern."""


class SandboxExecutionError(Exception):
    """Raised when the LLM-generated code crashes during execution."""


def screen_code(code: str) -> None:
    """Reject LLM code that contains any forbidden pattern.

    Called BEFORE compile/exec so dangerous code never enters the
    interpreter.
    """
    if not isinstance(code, str):
        raise SandboxRejected(f"code must be str, got {type(code).__name__}")
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern in code:
            raise SandboxRejected(
                f"sandbox rejected: forbidden pattern {pattern!r} present in LLM code"
            )


def run_analysis(
    code: str,
    *,
    trades_df: Any,
    ohlcv: dict,
    extra_globals: dict | None = None,
) -> list[dict]:
    """Execute LLM-generated analysis code and return the ``result`` variable.

    Args:
        code: The Python source the LLM produced.
        trades_df: A pandas DataFrame of recent closed trades, exposed
            in the sandbox as ``trades_df``.
        ohlcv: ``{symbol: {timeframe: pandas.DataFrame}}`` mapping,
            exposed as ``ohlcv``. May be empty.
        extra_globals: Additional names to bind into the sandbox.
            Tests use this to inject lightweight stand-ins.

    Returns:
        Whatever the LLM assigned to ``result``, which MUST be a list
        of dicts.

    Raises:
        SandboxRejected: code contained a forbidden pattern.
        SandboxExecutionError: code raised during execution OR did not
            produce a usable ``result`` variable.
    """
    screen_code(code)

    # Lazy imports — keeps the agent module-level import cheap and
    # makes the sandbox testable without scipy/statsmodels installed.
    pd = _try_import("pandas")
    np = _try_import("numpy")
    stats = _try_import_attr("scipy.stats", None)
    multipletests = _try_import_attr("statsmodels.stats.multitest", "multipletests")

    sandbox_globals: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "trades_df": trades_df,
        "ohlcv": dict(ohlcv or {}),
        "pd": pd,
        "np": np,
        "stats": stats,
        "multipletests": multipletests,
        # `result` is the variable the LLM is required to set.
        "result": None,
    }
    if extra_globals:
        sandbox_globals.update(extra_globals)

    try:
        compiled = compile(code, "<llm_analysis>", "exec")
    except SyntaxError as e:
        raise SandboxExecutionError(f"LLM code failed to compile: {e}") from e

    try:
        exec(compiled, sandbox_globals)  # noqa: S102 - this is the controlled sandbox
    except Exception as e:
        raise SandboxExecutionError(f"LLM code raised: {type(e).__name__}: {e}") from e

    result = sandbox_globals.get("result")
    if not isinstance(result, list):
        raise SandboxExecutionError(
            f"LLM code did not assign a list to `result` (got {type(result).__name__})"
        )
    for i, item in enumerate(result):
        if not isinstance(item, dict):
            raise SandboxExecutionError(
                f"`result[{i}]` is {type(item).__name__}, expected dict"
            )
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _try_import(name: str):
    """Try to import a module, returning None instead of raising."""
    try:
        return __import__(name)
    except ImportError:
        logger.debug(f"sandbox: optional dependency {name!r} not installed")
        return None


def _try_import_attr(module: str, attr: str | None):
    """Try to import a module + optionally pull a single attribute off it."""
    try:
        mod = __import__(module, fromlist=[attr] if attr else [])
    except ImportError:
        logger.debug(f"sandbox: optional dependency {module!r} not installed")
        return None
    if attr is None:
        return mod
    return getattr(mod, attr, None)

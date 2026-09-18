"""Stable release and build identity for API, workers, and clients."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PROJECT_ROOT / "VERSION"
API_VERSION = os.getenv("TIANSHU_API_VERSION", "2.0.0")


def _clean(value: Optional[str]) -> Optional[str]:
    value = value.strip() if value else ""
    return value or None


def _run_git(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _clean(result.stdout)


def _release_version() -> str:
    injected = _clean(os.getenv("TIANSHU_RELEASE_VERSION"))
    if injected:
        return injected
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip() or "0.0.0+unknown"
    except OSError:
        return "0.0.0+unknown"


def _git_commit() -> str:
    return _clean(os.getenv("TIANSHU_GIT_COMMIT")) or _run_git("rev-parse", "HEAD") or "unknown"


def _git_dirty() -> Optional[bool]:
    injected = _clean(os.getenv("TIANSHU_GIT_DIRTY"))
    if injected is not None:
        normalized = injected.lower()
        if normalized in {"1", "true", "yes", "dirty"}:
            return True
        if normalized in {"0", "false", "no", "clean"}:
            return False
        return None
    status = _run_git("status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


def _build_time() -> Optional[str]:
    return _clean(os.getenv("TIANSHU_BUILD_TIME")) or _run_git("show", "-s", "--format=%cI", "HEAD")


@lru_cache(maxsize=1)
def get_version_info() -> dict:
    """Return an immutable-per-process build fingerprint."""
    release_version = _release_version()
    git_commit = _git_commit()
    git_dirty = _git_dirty()
    short_commit = git_commit[:12] if git_commit != "unknown" else "unknown"
    build_id = f"{release_version}+{short_commit}"
    if git_dirty is True:
        build_id += ".dirty"
    return {
        "service": "mineru-tianshu",
        "api_version": API_VERSION,
        "release_version": release_version,
        "build_id": build_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "build_time": _build_time(),
    }


VERSION_INFO = get_version_info()
RELEASE_VERSION = VERSION_INFO["release_version"]
BUILD_ID = VERSION_INFO["build_id"]

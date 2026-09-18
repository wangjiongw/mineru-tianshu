import asyncio
import sys
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import version_info


def test_version_file_is_the_release_source_and_matches_package_metadata():
    release = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    metadata = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"(?ms)^\[project\].*?^version = \"([^\"]+)\"", metadata)

    assert release == "2.1.0"
    assert match and match.group(1) == release


def test_injected_build_metadata_produces_stable_build_id(monkeypatch):
    monkeypatch.setenv("TIANSHU_RELEASE_VERSION", "2.1.3")
    monkeypatch.setenv("TIANSHU_GIT_COMMIT", "abcdef1234567890")
    monkeypatch.setenv("TIANSHU_GIT_DIRTY", "false")
    monkeypatch.setenv("TIANSHU_BUILD_TIME", "2026-09-18T10:00:00Z")
    version_info.get_version_info.cache_clear()

    result = version_info.get_version_info()

    assert result == {
        "service": "mineru-tianshu",
        "api_version": "2.0.0",
        "release_version": "2.1.3",
        "build_id": "2.1.3+abcdef123456",
        "git_commit": "abcdef1234567890",
        "git_dirty": False,
        "build_time": "2026-09-18T10:00:00Z",
    }
    version_info.get_version_info.cache_clear()


def test_dirty_build_is_explicit_in_build_id(monkeypatch):
    monkeypatch.setenv("TIANSHU_RELEASE_VERSION", "2.0.0")
    monkeypatch.setenv("TIANSHU_GIT_COMMIT", "1234567890abcdef")
    monkeypatch.setenv("TIANSHU_GIT_DIRTY", "true")
    monkeypatch.setenv("TIANSHU_BUILD_TIME", "2026-09-18T10:00:00Z")
    version_info.get_version_info.cache_clear()

    result = version_info.get_version_info()

    assert result["git_dirty"] is True
    assert result["build_id"] == "2.0.0+1234567890ab.dirty"
    version_info.get_version_info.cache_clear()


def test_api_root_version_and_health_use_same_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("AUTH_DATABASE_PATH", str(tmp_path / "auth.db"))
    import api_server

    expected = {
        "service": "mineru-tianshu",
        "api_version": "2.0.0",
        "release_version": "2.1.0",
        "build_id": "2.1.0+abcdef123456",
        "git_commit": "abcdef1234567890",
        "git_dirty": False,
        "build_time": "2026-09-18T10:00:00Z",
    }

    class FakeDB:
        @staticmethod
        def get_queue_stats():
            return {"pending": 0}

    monkeypatch.setattr(api_server, "get_version_info", lambda: expected)
    monkeypatch.setattr(api_server, "db", FakeDB())

    root = asyncio.run(api_server.root())
    assert api_server.version_info() == expected
    assert root["build"] == expected
    assert api_server.health_check()["version"] == expected
    assert api_server.app.version == "2.1.0"


def test_worker_health_payload_includes_shared_version_signal():
    source = (BACKEND / "litserve_worker.py").read_text(encoding="utf-8")
    assert 'from version_info import get_version_info' in source
    assert '"version": get_version_info()' in source

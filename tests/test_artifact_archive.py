import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from artifact_archive import create_task_artifact_archive


def test_create_task_artifact_archive_preserves_relative_tree(tmp_path):
    result_dir = tmp_path / "result"
    nested = result_dir / "chunks" / "0001_pages_1-2"
    nested.mkdir(parents=True)
    (result_dir / "result.md").write_text("markdown", encoding="utf-8")
    (nested / "layout.pdf").write_bytes(b"layout")

    archive_path, download_name = create_task_artifact_archive(
        result_dir,
        "task-id",
        "paper name.pdf",
    )
    try:
        assert download_name == "paper_name_mineru_results.zip"
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.namelist() == [
                "paper_name_mineru_results/chunks/0001_pages_1-2/layout.pdf",
                "paper_name_mineru_results/result.md",
            ]
            assert (
                archive.read("paper_name_mineru_results/result.md")
                == b"markdown"
            )
    finally:
        archive_path.unlink(missing_ok=True)

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


def test_canonical_archive_contains_structured_alias_and_referenced_images_only(tmp_path):
    result_dir = tmp_path / "result"
    images = result_dir / "images"
    images.mkdir(parents=True)
    (result_dir / "full.md").write_text(
        "![used](images/used.png)\n<img src=\"images/also-used.jpg\">",
        encoding="utf-8",
    )
    (result_dir / "result.json").write_text("[]\n", encoding="utf-8")
    (result_dir / "result.md").write_text("online markdown", encoding="utf-8")
    (result_dir / "result_content_list.json").write_text("[]\n", encoding="utf-8")
    (result_dir / "result_content_list_v2.json").write_text("[[]]\n", encoding="utf-8")
    (result_dir / "mineru_model.json").write_text("[{\"private\": true}]", encoding="utf-8")
    (result_dir / "source.pdf").write_bytes(b"%PDF")
    (images / "used.png").write_bytes(b"used")
    (images / "also-used.jpg").write_bytes(b"also")
    (images / "unused.png").write_bytes(b"unused")

    archive_path, _ = create_task_artifact_archive(
        result_dir, "task-id", "paper.pdf", profile="canonical"
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            suffixes = {
                Path(name).relative_to(Path(name).parts[0]).as_posix()
                for name in archive.namelist()
            }
            assert suffixes == {
                "result_content_list.json",
                "result_content_list_v2.json",
                "content_list.json",
                "full.md",
                "images/also-used.jpg",
                "images/used.png",
                "result.md",
                "mineru_model.json",
                "result.json",
            }
            root = archive.namelist()[0].split("/", 1)[0]
            assert archive.read(f"{root}/content_list.json") == archive.read(f"{root}/result.json")
    finally:
        archive_path.unlink(missing_ok=True)

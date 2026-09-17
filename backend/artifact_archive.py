"""Build temporary ZIP archives for completed task artifacts."""

import os
import re
import tempfile
import zipfile
from pathlib import Path


def create_task_artifact_archive(
    result_dir: Path,
    task_id: str,
    file_name: str | None,
) -> tuple[Path, str]:
    """Archive every regular file below the result directory."""
    root = result_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {root}")

    stem = re.sub(
        r"[^\w.-]+",
        "_",
        Path(file_name or "").stem,
        flags=re.UNICODE,
    ).strip("._") or task_id
    archive_root = f"{stem}_mineru_results"
    file_descriptor, archive_name = tempfile.mkstemp(
        prefix=f"mineru_{task_id}_",
        suffix=".zip",
    )
    os.close(file_descriptor)
    archive_path = Path(archive_name)

    try:
        file_count = 0
        with zipfile.ZipFile(
            archive_path,
            "w",
            zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            for item in sorted(root.rglob("*")):
                if item.is_symlink() or not item.is_file():
                    continue
                resolved = item.resolve()
                if not resolved.is_relative_to(root):
                    continue
                archive.write(
                    resolved,
                    str(Path(archive_root) / resolved.relative_to(root)),
                )
                file_count += 1
        if not file_count:
            raise ValueError("Result directory contains no downloadable files")
        return archive_path, f"{archive_root}.zip"
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

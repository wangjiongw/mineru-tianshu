"""Build temporary ZIP archives for completed task artifacts."""

import os
import re
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit


CANONICAL_PROFILE = "canonical"
_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(([^)]+)\)|<img[^>]+src=[\"']([^\"']+)",
    re.IGNORECASE,
)


def _canonical_members(root: Path) -> tuple[list[tuple[Path, Path]], bytes | None]:
    """Resolve the compact, consumer-facing artifact contract."""
    markdown = root / "full.md"
    if not markdown.is_file():
        raise ValueError("Canonical artifact profile requires full.md")

    result_json = root / "result.json"
    content_list = root / "content_list.json"
    if not result_json.is_file() and content_list.is_file():
        result_json = content_list
    if not result_json.is_file():
        raise ValueError("Canonical artifact profile requires result.json")

    members: list[tuple[Path, Path]] = [
        (markdown, Path("full.md")),
        (result_json, Path("result.json")),
    ]
    alias_bytes = None
    if content_list.is_file():
        members.append((content_list, Path("content_list.json")))
    else:
        alias_bytes = result_json.read_bytes()
    for name in ("result.md", "mineru_model.json"):
        artifact = root / name
        if artifact.is_file():
            members.append((artifact, Path(name)))

    # MinerU-specific public artifacts are additive. Generic consumers keep
    # using result.json while MinerU clients receive the official v1/v2 files.
    for pattern in ("*_content_list.json", "*_content_list_v2.json"):
        for artifact in sorted(root.glob(pattern)):
            members.append((artifact, Path(artifact.name)))

    manifest = root / "merge_manifest.json"
    if manifest.is_file():
        members.append((manifest, Path("merge_manifest.json")))

    markdown_text = markdown.read_text(encoding="utf-8")
    referenced: set[Path] = set()
    for left, right in _MARKDOWN_IMAGE_RE.findall(markdown_text):
        value = (left or right).strip().strip("<>")
        parsed = urlsplit(value)
        if parsed.scheme == "data":
            continue
        if parsed.scheme or parsed.netloc or parsed.path.startswith(("/api/", "/v1/")):
            raise ValueError(f"Canonical full.md contains a remote image reference: {value}")
        relative = Path(unquote(parsed.path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Canonical full.md contains an unsafe image reference: {value}")
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Canonical full.md references a missing image: {value}")
        referenced.add(relative)

    members.extend(((root / relative).resolve(), relative) for relative in sorted(referenced))
    return members, alias_bytes


def create_task_artifact_archive(
    result_dir: Path,
    task_id: str,
    file_name: str | None,
    profile: str = "all",
) -> tuple[Path, str]:
    """Archive task files using the full or canonical artifact profile."""
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
        if profile not in {"all", CANONICAL_PROFILE}:
            raise ValueError(f"Unsupported artifact profile: {profile}")
        canonical_members = None
        alias_bytes = None
        if profile == CANONICAL_PROFILE:
            canonical_members, alias_bytes = _canonical_members(root)

        file_count = 0
        with zipfile.ZipFile(
            archive_path,
            "w",
            zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            items = canonical_members or [
                (item, item.relative_to(root))
                for item in sorted(root.rglob("*"))
            ]
            for item, relative in items:
                if item.is_symlink() or not item.is_file():
                    continue
                resolved = item.resolve()
                if not resolved.is_relative_to(root):
                    continue
                archive.write(
                    resolved,
                    str(Path(archive_root) / relative),
                )
                file_count += 1
            if alias_bytes is not None:
                archive.writestr(str(Path(archive_root) / "content_list.json"), alias_bytes)
                file_count += 1
        if not file_count:
            raise ValueError("Result directory contains no downloadable files")
        return archive_path, f"{archive_root}.zip"
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

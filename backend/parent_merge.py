import filecmp
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote, urlsplit


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}


class ParentMergeInputError(ValueError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def _task_options(task: dict) -> dict:
    try:
        options = task.get("options") or {}
        return json.loads(options) if isinstance(options, str) else dict(options)
    except Exception:
        return {}


def child_start_page(child: dict) -> int:
    try:
        return int(_task_options(child).get("chunk_info", {}).get("start_page", 0))
    except Exception:
        return 0


def child_page_count(child: dict) -> Optional[int]:
    try:
        value = _task_options(child).get("chunk_info", {}).get("page_count")
        return int(value) if value is not None else None
    except Exception:
        return None


def count_result_pages(result: dict) -> Optional[int]:
    value = result.get("total_pages")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    json_content = result.get("json_content")
    if isinstance(json_content, dict):
        pages = json_content.get("pages")
        if isinstance(pages, list):
            return len(pages)
    if isinstance(json_content, list):
        return len(json_content)
    return None


def build_completion_payload(
    result: dict,
    *,
    page_count: Optional[int],
    worker_group_index: Optional[int],
    worker_child_index: Optional[int],
    processing_seconds: Optional[float],
) -> str:
    payload = {
        "pdf_path": result.get("pdf_path"),
        "json_content": result.get("json_content"),
        "markdown": result.get("content"),
        "markdown_file": result.get("markdown_file"),
        "metrics": {
            "page_count": page_count,
            "worker_group_index": worker_group_index,
            "worker_child_index": worker_child_index,
            "processing_seconds": processing_seconds,
        },
    }
    return json.dumps(payload)


def _child_result_dir(child: dict) -> Optional[Path]:
    result_path = child.get("result_path")
    if not result_path:
        return None
    result_dir = Path(result_path)
    return result_dir if result_dir.is_dir() else None


def find_child_markdown(child: dict) -> Optional[Path]:
    """Prefer offline-safe full.md over result.md with child-specific URLs."""
    result_dir = _child_result_dir(child)
    if result_dir is None:
        return None
    for name in ("full.md", "result.md"):
        candidate = result_dir / name
        if candidate.is_file():
            return candidate
    markdown_files = sorted(result_dir.rglob("*.md"))
    return markdown_files[0] if markdown_files else None


def _find_child_content_list(child: dict) -> Optional[Path]:
    result_dir = _child_result_dir(child)
    if result_dir is None:
        return None
    for name in ("result.json", "content_list.json"):
        candidate = result_dir / name
        if candidate.is_file():
            return candidate
    candidates = sorted(
        path
        for path in result_dir.rglob("*.json")
        if "content_list" in path.name and "_v2" not in path.name
    )
    return candidates[0] if candidates else None


def _find_child_model(child: dict) -> Optional[Path]:
    result_dir = _child_result_dir(child)
    if result_dir is None:
        return None
    candidate = result_dir / "mineru_model.json"
    if candidate.is_file():
        return candidate
    candidates = sorted(result_dir.rglob("*_model.json"))
    return candidates[0] if candidates else None


def _is_mineru_parent(parent_task: dict) -> bool:
    backend = str(parent_task.get("backend") or "").lower()
    return (
        backend == "auto"
        or "pipeline" in backend
        or backend.startswith("hybrid-")
        or backend.startswith("vlm-")
    )


def validate_parent_merge_inputs(parent_task: dict) -> tuple[list[dict], list[str]]:
    if not parent_task:
        return [], ["parent task not found"]
    parent_task_id = parent_task.get("task_id", "<unknown>")
    child_count = int(parent_task.get("child_count") or 0)
    children = parent_task.get("children", [])
    completed_children = [child for child in children if child.get("status") == "completed"]
    reasons = []
    if len(children) != child_count:
        reasons.append(
            f"child row count mismatch for {parent_task_id}: rows={len(children)} expected={child_count}"
        )
    if len(completed_children) != child_count:
        reasons.append(
            f"completed child count mismatch for {parent_task_id}: completed={len(completed_children)} expected={child_count}"
        )

    require_mineru_core = _is_mineru_parent(parent_task)
    content_presence = []
    model_presence = []
    for child in completed_children:
        child_id = child.get("task_id", "<unknown>")
        result_path = child.get("result_path")
        if not result_path:
            reasons.append(f"child {child_id} has no result_path")
            continue
        result_dir = Path(result_path)
        if not result_dir.exists():
            reasons.append(f"child {child_id} result_path missing: {result_path}")
            continue
        if not result_dir.is_dir():
            reasons.append(f"child {child_id} result_path is not a directory: {result_path}")
            continue
        if find_child_markdown(child) is None:
            reasons.append(f"child {child_id} has no markdown artifact under {result_path}")
        content_presence.append((child_id, _find_child_content_list(child)))
        model_presence.append((child_id, _find_child_model(child)))

    require_content = require_mineru_core or any(path for _child_id, path in content_presence)
    require_model = require_mineru_core or any(path for _child_id, path in model_presence)
    if require_content:
        reasons.extend(
            f"child {child_id} has no content-list artifact"
            for child_id, path in content_presence
            if path is None
        )
    if require_model:
        reasons.extend(
            f"child {child_id} has no mineru-model artifact"
            for child_id, path in model_presence
            if path is None
        )
    return completed_children, reasons


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_tree_link_first(source: Path, destination: Path) -> None:
    for item in sorted(source.rglob("*")):
        if item.is_symlink() or not item.is_file():
            continue
        _link_or_copy(item, destination / item.relative_to(source))


def _same_file_contents(left: Path, right: Path) -> bool:
    return left.stat().st_size == right.stat().st_size and filecmp.cmp(left, right, shallow=False)


def _copy_child_images(child: dict, image_dir: Path) -> dict[str, str]:
    result_dir = _child_result_dir(child)
    source_dir = result_dir / "images" if result_dir else None
    if source_dir is None or not source_dir.is_dir():
        return {}

    image_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    chunk_label = f"p{max(child_start_page(child), 1)}"
    for source in sorted(source_dir.iterdir()):
        if source.is_symlink() or not source.is_file():
            continue
        destination = image_dir / source.name
        if destination.exists() and not _same_file_contents(source, destination):
            destination = image_dir / f"{chunk_label}_{source.name}"
            counter = 1
            while destination.exists() and not _same_file_contents(source, destination):
                destination = image_dir / f"{chunk_label}_{counter}_{source.name}"
                counter += 1
        if not destination.exists():
            _link_or_copy(source, destination)
        mapping[source.name] = destination.name
    return mapping


def _mapped_image_name(value: str, mapping: dict[str, str]) -> Optional[str]:
    if not isinstance(value, str) or not mapping:
        return None
    try:
        path = urlsplit(value).path
    except Exception:
        path = value
    name = Path(path).name
    if name not in mapping or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return mapping[name]


def _rewrite_json_image_paths(value, mapping: dict[str, str]):
    if isinstance(value, dict):
        return {key: _rewrite_json_image_paths(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_json_image_paths(item, mapping) for item in value]
    if isinstance(value, str):
        mapped = _mapped_image_name(value, mapping)
        if mapped and ("images/" in value or "/images/" in value):
            return f"images/{mapped}"
    return value


def _rewrite_markdown_image_paths(content: str, mapping: dict[str, str]) -> str:
    def replace_markdown(match):
        mapped = _mapped_image_name(match.group(2), mapping)
        return f"![{match.group(1)}](images/{mapped})" if mapped else match.group(0)

    def replace_html(match):
        mapped = _mapped_image_name(match.group(2), mapping)
        if not mapped:
            return match.group(0)
        return f'<img{match.group(1)}src="images/{mapped}"{match.group(3)}>'

    content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_markdown, content)
    return re.sub(r'<img([^>]*?)src=["\']([^"\']+)["\']([^>]*)>', replace_html, content)


def _offset_page_indexes(value, offset: int):
    if isinstance(value, dict):
        return {
            key: item + offset if key == "page_idx" and isinstance(item, int) else _offset_page_indexes(item, offset)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_offset_page_indexes(item, offset) for item in value]
    return value


def _content_items_from_child(child: dict, image_mapping: dict[str, str]) -> list:
    content_file = _find_child_content_list(child)
    if content_file is None:
        return []
    try:
        data = json.loads(content_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ParentMergeInputError([f"invalid content list for child {child.get('task_id')}: {exc}"]) from exc
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("pages"), list):
        items = data["pages"]
    else:
        raise ParentMergeInputError([f"unsupported content-list shape for child {child.get('task_id')}"])
    items = _rewrite_json_image_paths(items, image_mapping)
    return _offset_page_indexes(items, max(child_start_page(child), 1) - 1)


def _model_pages_from_child(child: dict, image_mapping: dict[str, str]) -> list:
    model_file = _find_child_model(child)
    if model_file is None:
        return []
    try:
        pages = json.loads(model_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ParentMergeInputError([f"invalid model JSON for child {child.get('task_id')}: {exc}"]) from exc
    if not isinstance(pages, list):
        raise ParentMergeInputError([f"mineru model for child {child.get('task_id')} is not a page list"])
    expected = child_page_count(child)
    if expected is not None and len(pages) != expected:
        raise ParentMergeInputError(
            [f"mineru model page count mismatch for child {child.get('task_id')}: pages={len(pages)} expected={expected}"]
        )
    return _rewrite_json_image_paths(pages, image_mapping)


def _api_markdown(local_markdown: str, parent_out: Path, output_dir: str) -> str:
    try:
        relative = parent_out.relative_to(Path(output_dir))
    except ValueError:
        relative = Path(parent_out.name)

    def api_url(name: str) -> str:
        path = str(relative / "images" / name).replace("\\", "/")
        return f"/api/v1/files/output/{quote(path, safe='/')}"

    def replace_markdown(match):
        name = Path(urlsplit(match.group(2)).path).name
        return f"![{match.group(1)}]({api_url(name)})" if name else match.group(0)

    def replace_html(match):
        name = Path(urlsplit(match.group(2)).path).name
        return f'<img{match.group(1)}src="{api_url(name)}"{match.group(3)}>' if name else match.group(0)

    content = re.sub(r"!\[([^\]]*)\]\((images/[^)]+)\)", replace_markdown, local_markdown)
    return re.sub(r'<img([^>]*?)src=["\'](images/[^"\']+)["\']([^>]*)>', replace_html, content)


def _validate_local_image_references(markdown: str, json_values: list, image_dir: Path) -> None:
    references = [
        left or right
        for left, right in re.findall(
            r"!\[[^\]]*\]\((images/[^)]+)\)|<img[^>]+src=[\"'](images/[^\"']+)", markdown
        )
    ]

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str) and value.startswith("images/"):
            references.append(value)

    collect(json_values)
    missing = sorted({ref for ref in references if not (image_dir / Path(ref).name).is_file()})
    if missing:
        raise ParentMergeInputError([f"merged output has missing image references: {', '.join(missing[:10])}"])


def _copy_parent_pdf(
    parent_task: dict,
    staging_dir: Path,
    existing_parent: Path,
    ensure_pdf_in_output: Optional[Callable[[str, Path], object]],
) -> None:
    if ensure_pdf_in_output:
        ensure_pdf_in_output(parent_task["file_path"], staging_dir)
    else:
        source_pdf = Path(parent_task["file_path"])
        if source_pdf.is_file() and source_pdf.suffix.lower() == ".pdf":
            shutil.copy2(source_pdf, staging_dir / source_pdf.name)
    if not list(staging_dir.glob("*.pdf")) and existing_parent.is_dir():
        existing_pdf = next((path for path in sorted(existing_parent.glob("*.pdf")) if path.is_file()), None)
        if existing_pdf:
            _link_or_copy(existing_pdf, staging_dir / existing_pdf.name)
    if str(parent_task.get("file_name") or "").lower().endswith(".pdf") and not list(staging_dir.glob("*.pdf")):
        raise ParentMergeInputError([f"parent source PDF is unavailable for {parent_task.get('task_id')}"])


def _promote_staging(staging_dir: Path, parent_out: Path) -> None:
    backup = parent_out.parent / f".{parent_out.name}.previous-{uuid.uuid4().hex}"
    moved_existing = False
    try:
        if parent_out.exists():
            os.replace(parent_out, backup)
            moved_existing = True
        os.replace(staging_dir, parent_out)
    except Exception:
        if moved_existing and backup.exists() and not parent_out.exists():
            os.replace(backup, parent_out)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def build_parent_task_artifacts(
    *,
    parent_task: dict,
    output_dir: str,
    ensure_pdf_in_output: Optional[Callable[[str, Path], object]] = None,
) -> Path:
    completed_children, input_errors = validate_parent_merge_inputs(parent_task)
    if input_errors:
        raise ParentMergeInputError(input_errors)
    completed_children.sort(key=child_start_page)

    parent_out = Path(output_dir) / Path(parent_task["file_path"]).stem
    parent_out.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = parent_out.parent / f".{parent_out.name}.merge-{os.getpid()}-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True)
    preserve_all = bool(_task_options(parent_task).get("preserve_all_artifacts", False))

    try:
        markdown_parts = []
        content_items = []
        model_pages = []
        image_dir = staging_dir / "images"
        image_dir.mkdir()
        for index, child in enumerate(completed_children, start=1):
            image_mapping = _copy_child_images(child, image_dir)
            markdown_file = find_child_markdown(child)
            if markdown_file is None:
                raise ParentMergeInputError([f"child {child.get('task_id')} has no markdown artifact"])
            markdown_parts.append(
                _rewrite_markdown_image_paths(markdown_file.read_text(encoding="utf-8"), image_mapping)
            )
            content_items.extend(_content_items_from_child(child, image_mapping))
            model_pages.extend(_model_pages_from_child(child, image_mapping))

            if preserve_all:
                start = max(child_start_page(child), 1)
                count = child_page_count(child) or 0
                end = start + count - 1 if count else start
                chunk_dir = staging_dir / "chunks" / f"{index:04d}_pages_{start}-{end}"
                _copy_tree_link_first(Path(child["result_path"]), chunk_dir)

        merged_markdown = "\n\n\n\n".join(markdown_parts)
        _validate_local_image_references(merged_markdown, [content_items, model_pages], image_dir)
        (staging_dir / "full.md").write_text(merged_markdown, encoding="utf-8")
        (staging_dir / "result.md").write_text(
            _api_markdown(merged_markdown, parent_out, output_dir), encoding="utf-8"
        )
        has_content_lists = any(_find_child_content_list(child) for child in completed_children)
        has_model_outputs = any(_find_child_model(child) for child in completed_children)
        if has_content_lists:
            result_json = staging_dir / "result.json"
            result_json.write_text(json.dumps(content_items, indent=2, ensure_ascii=False), encoding="utf-8")
            _link_or_copy(result_json, staging_dir / "content_list.json")
        if has_model_outputs:
            (staging_dir / "mineru_model.json").write_text(
                json.dumps(model_pages, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        _copy_parent_pdf(parent_task, staging_dir, parent_out, ensure_pdf_in_output)
        _promote_staging(staging_dir, parent_out)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return parent_out


def rebuild_completed_parent_artifacts(*, task_db, parent_task_id: str, output_dir: str) -> Path:
    parent_task = task_db.get_task_with_children(parent_task_id)
    if not parent_task or parent_task.get("status") != "completed" or not parent_task.get("is_parent"):
        raise ParentMergeInputError([f"task {parent_task_id} is not a completed split parent"])
    return build_parent_task_artifacts(parent_task=parent_task, output_dir=output_dir)


def merge_parent_task_results(
    *,
    task_db,
    parent_task_id: str,
    output_dir: str,
    merge_owner: str,
    ensure_pdf_in_output: Optional[Callable[[str, Path], object]] = None,
    cleanup_child_files: bool = True,
) -> Path:
    parent_task = task_db.get_task_with_children(parent_task_id)
    parent_out = build_parent_task_artifacts(
        parent_task=parent_task,
        output_dir=output_dir,
        ensure_pdf_in_output=ensure_pdf_in_output,
    )
    if not task_db.complete_parent_merge(parent_task_id, str(parent_out), merge_owner):
        raise RuntimeError(f"Parent merge completion was not accepted for {parent_task_id}")

    if cleanup_child_files:
        cleanup_completed_child_files(parent_task["children"])
        cleanup_empty_split_dir(output_dir, parent_task_id)
    return parent_out


def cleanup_completed_child_files(children: list[dict]) -> None:
    for child in children:
        if child.get("status") != "completed":
            continue
        try:
            if child.get("file_path"):
                Path(child["file_path"]).unlink(missing_ok=True)
        except Exception:
            pass


def cleanup_empty_split_dir(output_dir: str, parent_task_id: str) -> None:
    try:
        split_dir = Path(output_dir) / "splits" / parent_task_id
        if split_dir.exists() and not any(split_dir.iterdir()):
            split_dir.rmdir()
    except Exception:
        pass

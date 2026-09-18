import filecmp
import hashlib
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


def child_end_page(child: dict) -> Optional[int]:
    try:
        info = _task_options(child).get("chunk_info", {})
        value = info.get("end_page")
        if value is not None:
            return int(value)
        start = int(info.get("start_page"))
        count = int(info.get("page_count"))
        return start + count - 1
    except Exception:
        return None


def child_index(child: dict) -> int:
    try:
        return int(_task_options(child).get("chunk_info", {}).get("index", 0))
    except Exception:
        return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _page_indexes(value) -> list[int]:
    indexes = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "page_idx":
                if isinstance(item, bool) or not isinstance(item, int):
                    indexes.append(item)
                else:
                    indexes.append(item)
            else:
                indexes.extend(_page_indexes(item))
    elif isinstance(value, list):
        for item in value:
            indexes.extend(_page_indexes(item))
    return indexes


def _load_content_list(child: dict) -> list:
    path = _find_child_content_list(child)
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("pages"), list):
        return data["pages"]
    raise ValueError("unsupported content-list shape")


def _load_content_list_v2(child: dict) -> list:
    path = _find_child_content_list_v2(child)
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(page, list) for page in data):
        raise ValueError("MinerU content-list v2 must be a list of page lists")
    expected = child_page_count(child)
    if expected is not None and len(data) != expected:
        raise ValueError(
            f"MinerU content-list v2 page count mismatch: pages={len(data)} expected={expected}"
        )
    return data


def _load_model_pages(child: dict) -> list:
    path = _find_child_model(child)
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("mineru model is not a page list")
    return data


def _validate_chunk_sequence(parent_task: dict, children: list[dict]) -> list[str]:
    reasons = []
    ordered = sorted(children, key=lambda child: (child_start_page(child), child_index(child)))
    previous_end = None
    for child in ordered:
        child_id = child.get("task_id", "<unknown>")
        start = child_start_page(child)
        end = child_end_page(child)
        count = child_page_count(child)
        if start <= 0 or end is None or count is None or count <= 0:
            reasons.append(f"child {child_id} has invalid chunk_info")
            continue
        if end - start + 1 != count:
            reasons.append(
                f"child {child_id} range/count mismatch: {start}-{end} count={count}"
            )
        if previous_end is not None and start != previous_end + 1:
            kind = "overlap" if start <= previous_end else "gap"
            reasons.append(
                f"child range {kind}: previous_end={previous_end} next_start={start}"
            )
        previous_end = end
        try:
            content = _load_content_list(child)
            indexes = _page_indexes(content)
            if any(isinstance(value, bool) or not isinstance(value, int) for value in indexes):
                reasons.append(f"child {child_id} has non-integer page_idx")
            else:
                if any(value < 0 or value >= count for value in indexes):
                    reasons.append(f"child {child_id} has page_idx outside 0-{count - 1}")
                if indexes != sorted(indexes):
                    reasons.append(f"child {child_id} has non-monotonic page_idx")
        except Exception as exc:
            reasons.append(f"invalid content list for child {child_id}: {exc}")
        try:
            model_pages = _load_model_pages(child)
            if len(model_pages) != count:
                reasons.append(
                    f"mineru model page count mismatch for child {child_id}: "
                    f"pages={len(model_pages)} expected={count}"
                )
        except Exception as exc:
            reasons.append(f"invalid mineru model for child {child_id}: {exc}")

    split_info = _task_options(parent_task).get("split_info") or {}
    if ordered and split_info:
        expected_start = int(split_info.get("start_page", 0)) + 1
        expected_end = int(split_info.get("end_page", expected_start - 1)) + 1
        expected_pages = int(split_info.get("processed_pages", expected_end - expected_start + 1))
        actual_pages = sum(child_page_count(child) or 0 for child in ordered)
        if child_start_page(ordered[0]) != expected_start:
            reasons.append(
                f"first child starts at {child_start_page(ordered[0])}, expected {expected_start}"
            )
        if child_end_page(ordered[-1]) != expected_end:
            reasons.append(
                f"last child ends at {child_end_page(ordered[-1])}, expected {expected_end}"
            )
        if actual_pages != expected_pages:
            reasons.append(f"merged page count {actual_pages}, expected {expected_pages}")
    return reasons


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
    model_content = result.get("mineru_model_content")
    if isinstance(model_content, list):
        return len(model_content)
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


def _find_child_content_list_v2(child: dict) -> Optional[Path]:
    result_dir = _child_result_dir(child)
    if result_dir is None:
        return None
    candidate = result_dir / "result_content_list_v2.json"
    if candidate.is_file():
        return candidate
    candidates = sorted(result_dir.rglob("*_content_list_v2.json"))
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
    content_v2_presence = []
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
        content_v2_presence.append((child_id, _find_child_content_list_v2(child)))
        model_presence.append((child_id, _find_child_model(child)))

    require_content = require_mineru_core or any(path for _child_id, path in content_presence)
    require_content_v2 = any(path for _child_id, path in content_v2_presence)
    require_model = require_mineru_core or any(path for _child_id, path in model_presence)
    if require_content_v2:
        reasons.extend(
            f"child {child_id} has no MinerU content-list v2 artifact"
            for child_id, path in content_v2_presence if path is None
        )
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
    if len(completed_children) == child_count and not any(
        "no content-list" in reason or "no mineru-model" in reason for reason in reasons
    ):
        reasons.extend(_validate_chunk_sequence(parent_task, completed_children))
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


def _copy_child_images(child: dict, image_dir: Path, referenced_names: set[str]) -> dict[str, str]:
    result_dir = _child_result_dir(child)
    source_dir = result_dir / "images" if result_dir else None
    if source_dir is None or not source_dir.is_dir():
        return {}

    image_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    chunk_label = f"p{max(child_start_page(child), 1)}"
    for source in sorted(source_dir.iterdir()):
        if source.is_symlink() or not source.is_file() or source.name not in referenced_names:
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


def _collect_image_names(markdown: str, json_values: list) -> set[str]:
    values = []
    for left, right in re.findall(
        r"!\[[^\]]*\]\(([^)]+)\)|<img[^>]+src=[\"']([^\"']+)", markdown
    ):
        values.append(left or right)

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str):
            values.append(value)

    collect(json_values)
    names = set()
    for value in values:
        if not isinstance(value, str) or not ("images/" in value or "/images/" in value):
            continue
        name = Path(urlsplit(value).path).name
        if name and Path(name).suffix.lower() in IMAGE_SUFFIXES:
            names.add(name)
    return names


def _write_json_items(handle, values: list, first: bool) -> bool:
    for value in values:
        if not first:
            handle.write(",\n")
        handle.write(json.dumps(value, ensure_ascii=False))
        first = False
    return first


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
    references = []
    invalid = []
    for left, right in re.findall(
        r"!\[[^\]]*\]\(([^)]+)\)|<img[^>]+src=[\"']([^\"']+)", markdown
    ):
        value = left or right
        if value.startswith("data:"):
            continue
        path = urlsplit(value).path
        if not path.startswith("images/"):
            invalid.append(value)
        else:
            references.append(path)

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str) and ("images/" in value or "/images/" in value):
            path = urlsplit(value).path
            if not path.startswith("images/"):
                invalid.append(value)
            else:
                references.append(path)

    collect(json_values)
    if invalid:
        raise ParentMergeInputError([
            "merged output contains non-local image references: " + ", ".join(sorted(set(invalid))[:10])
        ])
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
    lease_refresh: Optional[Callable[[], bool]] = None,
) -> Path:
    completed_children, input_errors = validate_parent_merge_inputs(parent_task)
    if input_errors:
        raise ParentMergeInputError(input_errors)
    completed_children.sort(key=lambda child: (child_start_page(child), child_index(child)))

    parent_out = Path(output_dir) / Path(parent_task["file_path"]).stem
    parent_out.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = parent_out.parent / f".{parent_out.name}.merge-{os.getpid()}-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True)
    preserve_all = bool(_task_options(parent_task).get("preserve_all_artifacts", False))
    has_content_lists = any(_find_child_content_list(child) for child in completed_children)
    has_content_lists_v2 = bool(completed_children) and all(
        _find_child_content_list_v2(child) for child in completed_children
    )
    has_model_outputs = any(_find_child_model(child) for child in completed_children)
    manifest = {
        "schema_version": 1,
        "task_id": parent_task.get("task_id"),
        "mode": "all" if preserve_all else "compact",
        "source": {},
        "boundary_policy": "strict-page-order-no-semantic-repair",
        "chunks": [],
        "totals": {"pages": 0, "content_items": 0, "markdown_characters": 0},
        "validation": {"status": "passed", "warnings": []},
        "outputs": {},
    }
    split_info = _task_options(parent_task).get("split_info") or {}
    manifest["source"].update({
        "total_pages": split_info.get("source_total_pages"),
        "processed_start_page": split_info.get("start_page"),
        "processed_end_page": split_info.get("end_page"),
    })
    if not split_info:
        manifest["validation"]["warnings"].append(
            "legacy parent has no split_info; range validated from child chunk_info only"
        )

    content_path = staging_dir / "result.json"
    content_v2_path = staging_dir / "result_content_list_v2.json"
    model_path = staging_dir / "mineru_model.json"
    content_handle = content_path.open("w", encoding="utf-8") if has_content_lists else None
    content_v2_handle = content_v2_path.open("w", encoding="utf-8") if has_content_lists_v2 else None
    model_handle = model_path.open("w", encoding="utf-8") if has_model_outputs else None
    content_first = True
    content_v2_first = True
    model_first = True
    image_dir = staging_dir / "images"
    image_dir.mkdir()

    try:
        if content_handle:
            content_handle.write("[\n")
        if content_v2_handle:
            content_v2_handle.write("[\n")
        if model_handle:
            model_handle.write("[\n")
        with (staging_dir / "full.md").open("w", encoding="utf-8") as full_md, \
             (staging_dir / "result.md").open("w", encoding="utf-8") as result_md:
            for index, child in enumerate(completed_children, start=1):
                markdown_file = find_child_markdown(child)
                if markdown_file is None:
                    raise ParentMergeInputError([f"child {child.get('task_id')} has no markdown artifact"])
                raw_markdown = markdown_file.read_text(encoding="utf-8")
                raw_content = _load_content_list(child) if has_content_lists else []
                raw_content_v2 = _load_content_list_v2(child) if has_content_lists_v2 else []
                raw_model = _load_model_pages(child) if has_model_outputs else []
                referenced_names = _collect_image_names(
                    raw_markdown, [raw_content, raw_content_v2, raw_model]
                )
                image_mapping = _copy_child_images(child, image_dir, referenced_names)

                markdown = _rewrite_markdown_image_paths(raw_markdown, image_mapping)
                content_items = _offset_page_indexes(
                    _rewrite_json_image_paths(raw_content, image_mapping),
                    max(child_start_page(child), 1) - 1,
                )
                content_v2_pages = _rewrite_json_image_paths(raw_content_v2, image_mapping)
                model_pages = _rewrite_json_image_paths(raw_model, image_mapping)
                _validate_local_image_references(
                    markdown, [content_items, content_v2_pages, model_pages], image_dir
                )

                if index > 1:
                    full_md.write("\n\n\n\n")
                    result_md.write("\n\n\n\n")
                full_md.write(markdown)
                result_md.write(_api_markdown(markdown, parent_out, output_dir))
                if content_handle:
                    content_first = _write_json_items(content_handle, content_items, content_first)
                if content_v2_handle:
                    content_v2_first = _write_json_items(
                        content_v2_handle, content_v2_pages, content_v2_first
                    )
                if model_handle:
                    model_first = _write_json_items(model_handle, model_pages, model_first)

                start_page = child_start_page(child)
                end_page = child_end_page(child)
                page_count = child_page_count(child) or 0
                manifest["chunks"].append({
                    "task_id": child.get("task_id"),
                    "index": child_index(child),
                    "start_page": start_page,
                    "end_page": end_page,
                    "page_count": page_count,
                    "model_pages": len(model_pages),
                    "content_items": len(content_items),
                    "content_v2_pages": len(content_v2_pages),
                    "markdown_characters": len(markdown),
                    "referenced_images": len(referenced_names),
                })
                manifest["totals"]["pages"] += page_count
                manifest["totals"]["content_items"] += len(content_items)
                manifest["totals"]["markdown_characters"] += len(markdown)

                if preserve_all:
                    chunk_dir = staging_dir / "chunks" / f"{index:04d}_pages_{start_page}-{end_page}"
                    _copy_tree_link_first(Path(child["result_path"]), chunk_dir)
                if lease_refresh is not None and not lease_refresh():
                    raise RuntimeError(f"lost parent merge lease for {parent_task.get('task_id')}")
        if content_handle:
            content_handle.write("\n]\n")
            content_handle.close()
            content_handle = None
            _link_or_copy(content_path, staging_dir / "content_list.json")
            _link_or_copy(content_path, staging_dir / "result_content_list.json")
        if content_v2_handle:
            content_v2_handle.write("\n]\n")
            content_v2_handle.close()
            content_v2_handle = None
        if model_handle:
            model_handle.write("\n]\n")
            model_handle.close()
            model_handle = None

        _copy_parent_pdf(parent_task, staging_dir, parent_out, ensure_pdf_in_output)
        source_pdf = next((path for path in sorted(staging_dir.glob("*.pdf")) if path.is_file()), None)
        if source_pdf:
            manifest["source"].update({
                "file": source_pdf.name,
                "sha256": _sha256(source_pdf),
                "bytes": source_pdf.stat().st_size,
            })
        manifest["totals"].update({
            "images": sum(1 for path in image_dir.iterdir() if path.is_file()),
            "image_bytes": sum(path.stat().st_size for path in image_dir.iterdir() if path.is_file()),
        })
        for name in (
            "full.md", "result.md", "result.json", "content_list.json",
            "result_content_list.json", "result_content_list_v2.json", "mineru_model.json",
        ):
            path = staging_dir / name
            if path.is_file():
                manifest["outputs"][name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        (staging_dir / "merge_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _promote_staging(staging_dir, parent_out)
    finally:
        if content_handle:
            content_handle.close()
        if content_v2_handle:
            content_v2_handle.close()
        if model_handle:
            model_handle.close()
        shutil.rmtree(staging_dir, ignore_errors=True)
    return parent_out

def rebuild_completed_parent_artifacts(*, task_db, parent_task_id: str, output_dir: str) -> Path:
    parent_task = task_db.get_task_with_children(parent_task_id)
    if not parent_task or parent_task.get("status") != "completed" or not parent_task.get("is_parent"):
        raise ParentMergeInputError([f"task {parent_task_id} is not a completed split parent"])
    parent_out = build_parent_task_artifacts(parent_task=parent_task, output_dir=output_dir)
    if not task_db.publish_completed_parent_result_path(parent_task_id, str(parent_out)):
        raise RuntimeError(f"completed parent result publication was not accepted for {parent_task_id}")
    return parent_out


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
        lease_refresh=(
            lambda: task_db.refresh_parent_merge_lease(parent_task_id, merge_owner)
            if hasattr(task_db, "refresh_parent_merge_lease") else True
        ),
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

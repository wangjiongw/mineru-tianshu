import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable, Optional

from output_normalizer import normalize_output


class ParentMergeInputError(ValueError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def child_start_page(child: dict) -> int:
    try:
        return int(json.loads(child.get("options") or "{}").get("chunk_info", {}).get("start_page", 0))
    except Exception:
        return 0


def child_page_count(child: dict) -> Optional[int]:
    try:
        value = json.loads(child.get("options") or "{}").get("chunk_info", {}).get("page_count")
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


def find_child_markdown(child: dict) -> Optional[Path]:
    result_path = child.get("result_path")
    if not result_path:
        return None
    res_dir = Path(result_path)
    if not res_dir.exists() or not res_dir.is_dir():
        return None
    md_file = next((f for f in res_dir.rglob("*.md") if f.name == "result.md"), None)
    if md_file is not None:
        return md_file
    md_files = list(res_dir.rglob("*.md"))
    return md_files[0] if md_files else None


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
    for child in completed_children:
        child_id = child.get("task_id", "<unknown>")
        result_path = child.get("result_path")
        if not result_path:
            reasons.append(f"child {child_id} has no result_path")
            continue
        res_dir = Path(result_path)
        if not res_dir.exists():
            reasons.append(f"child {child_id} result_path missing: {result_path}")
            continue
        if not res_dir.is_dir():
            reasons.append(f"child {child_id} result_path is not a directory: {result_path}")
            continue
        if find_child_markdown(child) is None:
            reasons.append(f"child {child_id} has no markdown artifact under {result_path}")
    return completed_children, reasons


def _json_pages_from_child(child: dict) -> list:
    res_dir = Path(child["result_path"])
    json_file = next((f for f in res_dir.rglob("*.json") if "result" in f.name or "content" in f.name), None)
    if not json_file:
        return []
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        offset = max(child_start_page(child), 1) - 1
        pages = data if isinstance(data, list) else data.get("pages", []) if isinstance(data, dict) else []
        for page in pages:
            if isinstance(page, dict) and "page_idx" in page:
                page["page_idx"] += offset
        return pages
    except Exception:
        return []


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
    completed_children, input_errors = validate_parent_merge_inputs(parent_task or {})
    if input_errors:
        raise ParentMergeInputError(input_errors)

    completed_children.sort(key=child_start_page)
    parent_out = Path(output_dir) / Path(parent_task["file_path"]).stem
    parent_out.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = parent_out.parent / f".{parent_out.name}.merge-{os.getpid()}-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True)

    try:
        md_parts = []
        json_pages = []
        for child in completed_children:
            md_file = find_child_markdown(child)
            if md_file is None:
                raise ParentMergeInputError([f"child {child.get('task_id', '<unknown>')} has no markdown artifact"])
            md_parts.append(md_file.read_text(encoding="utf-8"))
            json_pages.extend(_json_pages_from_child(child))

        (staging_dir / "result.md").write_text("\n\n\n\n".join(md_parts), encoding="utf-8")
        if json_pages:
            (staging_dir / "result.json").write_text(
                json.dumps(json_pages, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        parent_out.mkdir(parents=True, exist_ok=True)
        staged_names = {staged_file.name for staged_file in staging_dir.iterdir()}
        for stale_name in {"result.json"} - staged_names:
            stale_path = parent_out / stale_name
            if stale_path.exists():
                stale_path.unlink()
        for staged_file in staging_dir.iterdir():
            os.replace(staged_file, parent_out / staged_file.name)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    if ensure_pdf_in_output:
        ensure_pdf_in_output(parent_task["file_path"], parent_out)
    else:
        src_pdf = Path(parent_task["file_path"])
        if src_pdf.exists() and src_pdf.suffix.lower() == ".pdf":
            target = parent_out / src_pdf.name
            if not target.exists():
                shutil.copy2(src_pdf, target)

    normalize_output(parent_out)

    if not task_db.complete_parent_merge(parent_task_id, str(parent_out), merge_owner):
        raise RuntimeError(f"Parent merge completion was not accepted for {parent_task_id}")

    if cleanup_child_files:
        cleanup_completed_child_files(completed_children)
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

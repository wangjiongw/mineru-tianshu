import json
import sys
from pathlib import Path

import pikepdf

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Other worker tests intentionally install top-level import stubs at collection
# time. Load the real implementations without changing those shared stubs.
import importlib.util

_output_stub = sys.modules.pop("output_normalizer", None)
from output_normalizer.standard_output_normalizer import StandardOutputNormalizer
for _name in [name for name in sys.modules if name == "output_normalizer" or name.startswith("output_normalizer.")]:
    sys.modules.pop(_name, None)
if _output_stub is not None:
    sys.modules["output_normalizer"] = _output_stub

_pdf_spec = importlib.util.spec_from_file_location("real_pdf_utils", BACKEND / "utils" / "pdf_utils.py")
_pdf_module = importlib.util.module_from_spec(_pdf_spec)
_pdf_spec.loader.exec_module(_pdf_module)
split_pdf_file = _pdf_module.split_pdf_file


def test_split_pdf_validates_and_preserves_absolute_page_ranges(tmp_path):
    source = tmp_path / "source.pdf"
    with pikepdf.new() as pdf:
        for _ in range(53):
            pdf.add_blank_page(page_size=(100, 100))
        pdf.save(source)

    chunks = split_pdf_file(
        source, tmp_path / "chunks", chunk_size=20, parent_task_id="task",
        start_page=10, end_page=52,
    )

    assert [(item["start_page"], item["end_page"], item["page_count"]) for item in chunks] == [
        (11, 30, 20), (31, 50, 20), (51, 53, 3),
    ]
    for item in chunks:
        with pikepdf.open(item["path"]) as pdf:
            assert len(pdf.pages) == item["page_count"]


def test_compact_normalizer_preserves_mineru_v1_v2_and_referenced_images(tmp_path):
    output = tmp_path / "result"
    raw = output / "raw"
    images = output / "images"
    raw.mkdir(parents=True)
    images.mkdir()
    (images / "used.png").write_bytes(b"used")
    (images / "model-only.png").write_bytes(b"model")
    (images / "unused.png").write_bytes(b"unused")
    (raw / "document.md").write_text("![used](images/used.png)", encoding="utf-8")
    (raw / "doc_content_list.json").write_text(
        json.dumps([{"page_idx": 0, "text": "ok"}]), encoding="utf-8"
    )
    (raw / "doc_content_list_v2.json").write_text(
        json.dumps([[{"type": "paragraph", "content": {"paragraph_content": []}}]]), encoding="utf-8"
    )
    (raw / "doc_model.json").write_text(
        json.dumps([[{"img_path": "images/model-only.png"}]]), encoding="utf-8"
    )

    result = StandardOutputNormalizer(False, artifact_family="mineru")._normalize_local_files(output)

    assert json.loads((output / "result.json").read_text(encoding="utf-8"))[0]["text"] == "ok"
    assert (output / "content_list.json").read_bytes() == (output / "result.json").read_bytes()
    assert (output / "doc_content_list.json").read_bytes() == (output / "result.json").read_bytes()
    assert json.loads((output / "doc_content_list_v2.json").read_text(encoding="utf-8")) == [
        [{"type": "paragraph", "content": {"paragraph_content": []}}]
    ]
    assert json.loads((output / "mineru_model.json").read_text(encoding="utf-8"))[0][0]["img_path"] == "images/model-only.png"
    assert {path.name for path in images.iterdir()} == {"used.png", "model-only.png"}
    assert result["image_count"] == 2
    assert not (raw / "doc_content_list.json").exists()
    assert not (raw / "doc_content_list_v2.json").exists()
    assert not (raw / "doc_model.json").exists()

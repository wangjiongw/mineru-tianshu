import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import batch_parse_pdfs


class FakeResponse:
    def __init__(self, status, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}
        self.ok = 200 <= status < 300

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_submit_retries_503_and_reopens_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"pdf-data")
    responses = [FakeResponse(503, text="busy", headers={"Retry-After": "0"}), FakeResponse(200, {"task_id": "ok"})]
    bodies = []

    def post(*_args, **kwargs):
        bodies.append(kwargs["files"]["file"][1].read())
        return responses.pop(0)

    monkeypatch.setattr(batch_parse_pdfs.requests, "post", post)
    monkeypatch.setattr(batch_parse_pdfs.time, "sleep", lambda _delay: None)
    task_id = batch_parse_pdfs.submit_task("http://x", "t", pdf, "pipeline", "auto", "auto")
    assert task_id == "ok"
    assert bodies == [b"pdf-data", b"pdf-data"]


def test_submit_does_not_retry_unrelated_500(tmp_path, monkeypatch):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"pdf-data")
    calls = []

    def post(*_args, **_kwargs):
        calls.append(1)
        return FakeResponse(500, text="model crashed")

    monkeypatch.setattr(batch_parse_pdfs.requests, "post", post)
    try:
        batch_parse_pdfs.submit_task("http://x", "t", pdf, "pipeline", "auto", "auto")
    except Exception:
        pass
    else:
        raise AssertionError("unrelated HTTP 500 must be raised")
    assert len(calls) == 1

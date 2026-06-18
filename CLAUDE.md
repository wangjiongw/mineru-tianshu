# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MinerU Tianshu (天枢) is an enterprise-level AI data preprocessing platform that converts unstructured data into structured formats for AI applications. It supports documents, images, audio, video, and bioinformatics formats.

**Tech Stack:**
- Backend: FastAPI + Python 3.12+ + SQLite + Redis (optional)
- Frontend: Vue 3 + TypeScript + Vite + TailwindCSS + Pinia
- GPU/NPU Processing: LitServe for load balancing, PyTorch, PaddlePaddle
- Inference: **vLLM** serving the MinerU2.5-2B model (workers call it over HTTP)

**Two deployment targets (do not confuse them):**
- **8-card Ascend NPU bare metal** (`scripts/tianshu.sh`) — the active production environment for this repo. The `backend/kernel_meta/` directory is produced by CANN/torch_npu kernel compilation. vLLM runs as a separate service with 8 instances.
- **Docker** (`Makefile`, `docker-compose*.yml`) — the generic/portable path; builds target NVIDIA CUDA 12.6. Use this for non-NPU environments and the `cpu`/`offline`/`dev` compose variants.

## Data Paths (this environment)

Local (in-repo, used by the dev scripts):
- Test PDF input: `/data/projects/mineru/mineru-tianshu/input`
- Parsing output (dev): `/data/projects/mineru/mineru-tianshu/output`
- Analyze reports: `/data/projects/mineru/mineru-tianshu/docs`

Production NPU data store — **per-instance isolated**. Each instance gets its own subtree under a shared root:
- Data root: `DATA_ROOT=/share/wangjiong/databases/mineru_database`
- Per instance: `${DATA_ROOT}/${INSTANCE_ID}/{mineru_tianshu.db, mineru_outputs, mineru_uploads, mineru_logs}` where `INSTANCE_ID` defaults to `$(hostname)` (overridable via `export INSTANCE_ID=...`).
- Read-only assets stay shared (not isolated): `MODELSCOPE_CACHE`, `VLLM_MODEL_PATH`, code `PROJECT_ROOT`.
- Redis queue keys are namespaced per instance: `tianshu:task_queue:${INSTANCE_ID}`, `tianshu:processing:${INSTANCE_ID}`.

This prevents multi-instance collisions on the shared `/share` (SQLite has no WAL/file-lock — `task_db.py:76` — so two instances writing the same `.db` would deadlock; shared output/log dirs would clobber).

> **Instance conflict detection:** on `start`, `tianshu.sh` writes `${INSTANCE_DATA_DIR}/.instance.lock` containing the `INSTANCE_ID`. A different active instance already owning that dir aborts startup with a remediation hint; the same instance (e.g. pod restart, same hostname) rewrites its own lock. `stop all` releases it.

> **Legacy data (worker-0):** pre-isolation data lives at `/share/wangjiong/databases/mineru_verifier/` (13.7 GB, May 2026 snapshot; the `mineru/` sibling is a 15.8 GB Jun 2026 copy). Migrate worker-0 into the new layout with a same-volume `mv` (instant rename, no copy): `mkdir -p /share/wangjiong/databases/mineru_database && mv /share/wangjiong/databases/mineru_verifier /share/wangjiong/databases/mineru_database/"$(hostname)"`, then default `INSTANCE_ID=$(hostname)` picks it up. **Stop services before moving.**

## Common Commands

### Docker lifecycle (portable / CUDA / CPU)
```bash
make setup          # First-time deploy: create .env, build, start, show info
make build          # Build all images (parallel)
make start | stop | restart | down | status
make logs | logs-backend | logs-worker | logs-frontend
make dev            # Start docker-compose.dev.yml stack
make backup-db      # Copy SQLite out of the backend container into ./backups/
make clean          # DESTRUCTIVE: removes data/, logs/, models/ + volumes
```
`.env` is copied from `.env.example` on `setup`. Redis is auto-detected: set `REDIS_QUEUE_ENABLED=true` in `.env` and `make` adds the `--profile redis`.

### 8-card NPU bare metal (the active deployment)
```bash
# Unified lifecycle (VLLM → API → Workers → Frontend)
bash scripts/tianshu.sh start         # Start all services
bash scripts/tianshu.sh stop          # Stop all
bash scripts/tianshu.sh restart       # Restart all
bash scripts/tianshu.sh status        # Service overview
bash scripts/tianshu.sh logs worker   # Tail logs (vllm|api|worker|frontend|mcp)
bash scripts/tianshu.sh test          # End-to-end verification

# Per-service: start|stop accept redis|vllm|api|worker|frontend|mcp
bash scripts/tianshu.sh start redis
bash scripts/tianshu.sh stop worker
```
All paths, ports, vLLM model path, and instance counts are configured in the **config block at the top of `scripts/tianshu.sh`** (not in `.env`) — edit there to adapt to a new environment. Defaults: 8 vLLM instances (base port 30025), 8 workers (base port 8101, accelerator `cpu`), API 8000, MCP 8002, frontend 3000.

### Local development (no NPU/Docker) — preferred over `start_all.py`
```bash
bash scripts/dev-start.sh            # Interactive launcher (backend | frontend | both)
bash scripts/dev-backend.sh          # Backend only: sets env, creates test user, runs api_server.py on :8000
bash scripts/dev-frontend.sh         # Frontend only: npm dev on :3000 (proxies to :8000)
python scripts/init_dev_user.py      # Create/refresh the test admin user
```
The dev scripts point `DATABASE_PATH`/`OUTPUT_PATH`/`UPLOAD_PATH`/`MODEL_PATH` at the in-repo `data/` and `models/` dirs (created automatically). `dev-backend.sh` runs `init_dev_user.py` for you.

**Local test account:** `admin` / `admin123` (created by `init_dev_user.py`; login at `http://localhost:8000/api/v1/auth/login`).

Frontend dev flags: `FRONTEND_MODE=preview` builds then runs `npm run preview`; `VITE_OUT_DIR=<dir>` sets the build output directory (useful for multi-pod static deploys).

`start_all.py` (in `backend/`) is the full GPU-stack launcher (API + Worker + MCP together) — use it when you need the worker pool locally, not for light API-only work:
```bash
cd backend && python start_all.py --enable-mcp --devices 0,1 --workers-per-device 2
```

### Monitoring & ops helpers
```bash
python scripts/tianshu_status.py status                 # Service overview (queue/tasks/watch)
python scripts/tianshu_status.py tasks --status failed  # List tasks by status / search
python scripts/tianshu_status.py watch --interval 10    # Live monitor
# Auth via --username/--password, --token, or TIANSHU_TOKEN/TIANSHU_USER/TIANSHU_PASS env
python scripts/query_api_status.py                      # Probe API health, writes data/api_status.log
bash scripts/log_manager.sh list|view|tail|archive|clean|rotate   # Unified log management
```

### Linting & formatting
Python uses **Ruff** (`pyproject.toml`, target py312, line-length 120, `select=["E","F"]`, `ignore=["E402","E501"]`):
```bash
ruff check backend/      # Lint (--fix to auto-fix)
ruff format backend/     # Format
```
Frontend type-check is bundled into the build:
```bash
cd frontend && npm run build   # tsc && vite build
```
**Pre-commit** is configured (`.pre-commit-config.yaml`): Ruff format + lint on `backend/*.py`, shellcheck on `scripts/*.sh`, plus YAML/JSON/markdown/large-file checks. Install with `pip install pre-commit && pre-commit install`; run on everything with `pre-commit run --all-files`.

## Architecture

### Backend Structure
```
backend/
├── api_server.py           # FastAPI REST API server (port 8000)
├── litserve_worker.py      # GPU/NPU worker pool via LitServe (port 8001)
├── mcp_server.py           # MCP protocol server (port 8002)
├── task_db.py              # SQLite + Redis hybrid task queue (auth-scoped queries live here)
├── task_scheduler.py       # Task scheduling logic
├── redis_queue.py          # Redis queue implementation
├── start_all.py            # Launch API + Worker + MCP together
├── download_models.py      # Pull models on first use
├── auth/                   # JWT auth & authorization (auth_db, jwt_handler, routes, models)
├── mineru_pipeline/        # MinerU document processing
├── paddleocr_vl/           # PaddleOCR-VL engine (109+ languages)
├── paddleocr_vl_vllm/      # PaddleOCR-VL backed by the vLLM service instead of local weights
├── output_normalizer/      # Standardize engine outputs to common Markdown/JSON schema
├── audio_engines/          # Audio processing (SenseVoice)
├── video_engines/          # Video processing (FFmpeg + OCR)
├── format_engines/         # Bioinformatics formats (FASTA, GenBank)
├── remove_watermark/       # Watermark removal (experimental)
├── storage/                # RustFS object storage integration
└── utils/                  # Shared helpers (pdf_utils, perse_uitls)
```

### Frontend Structure
```
frontend/src/
├── api/        # API client (axios)
├── views/      # Page components (Dashboard, TaskSubmit, TaskList, TaskDetail, QueueManagement, ...)
├── components/ # Reusable components
├── stores/     # Pinia state management
├── router/     # Vue Router configuration
├── layouts/    # Layout components
├── locales/    # i18n translations (vue-i18n)
└── utils/      # Utility functions
```

### Service Architecture

**Services on the NPU deployment** (boot order matters — `tianshu.sh` enforces it; `start all` brings them up in this order):
1. **Redis** (6379) — task-queue cache; SQLite stays the source of truth. Brought up first so API/Workers find it on init.
2. **vLLM** — 8 instances of MinerU2.5-2B (base port 30025). The inference backend the workers call.
3. **API Server** (FastAPI, 8000) — HTTP requests, auth, task submission.
4. **Worker Pool** (LitServe, 8101+) — pull and process tasks across the 8 NPU cards.
5. **MCP Server** (optional, 8002) — Model Context Protocol for AI-assistant integration.
6. **Frontend** (3000) — Vue SPA.

**Task Flow:**
1. Client submits a task via the API (JWT bearer token).
2. Task is stored in SQLite (metadata) and optionally pushed to a Redis queue.
3. A worker claims the task via an atomic operation (prevents duplicate processing).
4. The worker routes the file to the right engine, calling vLLM for inference as needed.
5. Results are written, status updated, the client is notified.

**GPU/NPU Management:** LitServe handles device load balancing across cards; workers per device are configurable. Workers enter a "sleep" state when idle to free memory, with automatic CPU fallback when no accelerator is available.

## Key Concepts

### Authentication & Authorization
- **JWT tokens**: short-lived access + long-lived refresh. Default expiry in `tianshu.sh` is 30 days (`JWT_EXPIRE_MINUTES=43200`).
- **Roles**: `admin` (full access) and `user` (self-data only).
- **API keys**: users can generate API keys for external/scripted access.
- **Data isolation**: users can only access their own tasks, enforced in `task_db.py`.

### Task Processing Engines
- **pipeline**: MinerU standard document processing (PDF → Markdown/JSON).
- **paddleocr-vl** / **paddleocr-vl_vllm**: multi-language OCR (109+ languages); the `_vllm` variant offloads to the vLLM service.
- **sensevoice**: audio transcription with speaker diarization.
- **video_engine**: video processing (audio transcription + keyframe OCR).
- **format_engines**: bioinformatics formats (FASTA, GenBank).

### Task Deduplication & Shared Read
Tasks are deduplicated and support shared-read access (see commit `f54d843` and `docs/TASK_DEDUP_TEST.md`). Submitting identical content reuses prior results rather than reprocessing.

### PDF Auto-Splitting
Large PDFs (>threshold) are split into chunks for parallel processing:
- Controlled by `PDF_SPLIT_ENABLED`, `PDF_SPLIT_THRESHOLD_PAGES` (default 500), `PDF_SPLIT_CHUNK_SIZE` (default 500).
- Splitting runs **in the worker** (API stays fast); a parent task spawns child tasks processed in parallel, then results merge preserving original page numbers. ~40–60% faster on big files.

### RustFS Object Storage
Processed images are uploaded to RustFS (S3-compatible):
- Configured via `RUSTFS_PUBLIC_URL`, `RUSTFS_ACCESS_KEY`, `RUSTFS_SECRET_KEY`.
- Images in Markdown/JSON output are rewritten to public object-storage URLs.
- Organized by date: `YYYYMMDD/filename.ext`; filenames are timestamp-Base62 + NanoID.

## Configuration

### Environment Variables (`.env`)
Key options: `JWT_SECRET_KEY` (required, generate your own), `REDIS_QUEUE_ENABLED`, `MAX_FILE_SIZE`, `PDF_SPLIT_*`, `RUSTFS_*`, `WORKER_MEMORY_LIMIT`, `WORKER_MEMORY_RESERVATION`. Reference files: `.env.example` (Docker), `.env.cpu`, `backend/.env.example`.

> The NPU bare-metal run reads most runtime config from the **`tianshu.sh` config block**, not `.env`. `.env` governs the Docker path.

### Multi-instance isolation (`tianshu.sh`)
On the NPU path, `tianshu.sh` isolates each instance's writable data under `DATA_ROOT/<INSTANCE_ID>/` (see Data Paths above). Key config-block vars: `DATA_ROOT` (default `/share/wangjiong/databases/mineru_database`), `INSTANCE_ID` (default `$(hostname)`, override with `export INSTANCE_ID=...`), derived `INSTANCE_DATA_DIR` → `DATABASE_PATH`/`OUTPUT_PATH`/`UPLOAD_PATH`/`LOG_DIR`; Redis keys get an `:${INSTANCE_ID}` suffix. A `.instance.lock` guards against two different instances writing the same dir. **No backend change** — all paths/keys are read from env (`task_db.py:53`, `auth_db.py:39`, `api_server.py`, `litserve_worker.py:211`, `redis_queue.py:59-60`).

### Docker Compose Variants
`docker-compose.yml` (GPU/CUDA), `docker-compose.cpu.yml`, `docker-compose.dev.yml`, `docker-compose.offline.yml`.

## Further Reading (`docs/`)
Rich in-repo documentation — consult before deep-diving into source:
- `docs/DEPLOYMENT_GUIDE.md`, `docs/OPERATIONS.md`, `docs/SERVICE_MONITORING.md` — deploy/ops
- `docs/ARCHITECTURE_DIAGRAMS.md`, `docs/BACKEND_ARCHITECTURE.md`, `docs/FRONTEND_ARCHITECTURE.md`, `docs/DATA_WORKFLOW.md` — design & data flow
- `docs/API_REFERENCE.md` — full REST reference with curl + Python SDK examples
- `docs/SCRIPTS_REFERENCE.md`, `docs/logging_guide.md` — scripts & logs
- `docs/README.md` — documentation index by role

## Important Notes
1. **Concurrent task safety**: atomic DB/Redis operations prevent duplicate processing.
2. **GPU/NPU memory**: workers sleep when idle to free memory; CPU fallback when no accelerator.
3. **File cleanup**: old files are purged periodically (configurable).
4. **Offline deployment**: supported via pre-built images (`docker-compose.offline.yml`, `scripts/build-offline.sh`, `scripts/deploy-offline.sh`).
5. **Office documents**: `.doc/.docx/.pptx` are auto-converted to PDF via LibreOffice (in Docker).
6. **Dify integration**: `dify_plugin/` ships a Dify plugin (`parse_document_simple`) wrapping the platform.

## MCP Protocol Usage

When enabled, the MCP server (8002) exposes: `parse_document` (Base64 or URL, max 500MB), `get_task_status`, `list_tasks`, `get_queue_stats`.

**Claude Desktop / client config:**
```json
{
  "mcpServers": {
    "mineru-tianshu": {
      "url": "http://localhost:8002/sse",
      "transport": "sse"
    }
  }
}
```

## File Locations (local dev defaults)
- Database: `data/db/mineru_tianshu.db`
- Uploads: `data/uploads/`
- Output: `data/output/`
- Logs: `logs/backend/`, `logs/worker/`, `logs/mcp/` (NPU run writes to the `LOG_DIR` in `tianshu.sh`)
- Models: `models/` (auto-downloaded on first use)
- Docker volumes: defined in `docker-compose*.yml`

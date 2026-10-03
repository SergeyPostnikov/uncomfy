from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Literal

from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .database import get_session, init_db
from .models import Job
from .tasks import broker, run_job

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await init_db()
    await broker.startup()
    yield
    await broker.shutdown()


app = FastAPI(title="uncomfy", version="0.1.0", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

JobStatus = Literal["pending", "running", "done", "failed", "cancelled"]


class JobCreate(BaseModel):
    workflow: str = Field(default="img2img", description="Workflow name or 'img2img'")
    prompt: str | None = None
    params: dict[str, Any] | None = None


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    workflow: str
    prompt: str | None
    params: dict[str, Any] | None
    input_path: str | None
    output_path: str | None
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    duration_sec: float | None

    @classmethod
    def from_job(cls, job: Job) -> "JobResponse":
        return cls(
            id=job.id,
            status=job.status,
            workflow=job.workflow,
            prompt=job.prompt,
            params=job.params,
            input_path=job.input_path,
            output_path=job.output_path,
            error=job.error,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            finished_at=job.finished_at.isoformat() if job.finished_at else None,
            duration_sec=job.duration_sec,
        )


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

_UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)

_OUTPUT_DIR = Path(os.getenv("COMFY_PATH", r"C:\ComfyUI")) / "output"


async def _save_upload(file: UploadFile) -> Path:
    ext = Path(file.filename or "upload.png").suffix or ".png"
    dest = _UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    content = await file.read()
    dest.write_bytes(content)
    return dest


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/jobs", response_model=JobResponse, status_code=202)
async def create_job(
    workflow: str = Query(default="img2img"),
    prompt: str | None = Query(default=None),
    image: UploadFile | None = File(default=None),
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    """Create and enqueue a new inference job."""
    input_path: str | None = None
    if image is not None:
        saved = await _save_upload(image)
        input_path = str(saved)

    job = Job(
        workflow=workflow,
        prompt=prompt,
        input_path=input_path,
    )
    session.add(job)
    await session.commit()

    await run_job.kiq(job.id)
    logger.info("Enqueued job %s (workflow=%s)", job.id[:8], workflow)
    return JobResponse.from_job(job)


@app.post("/jobs/json", response_model=JobResponse, status_code=202)
async def create_job_json(
    body: JobCreate,
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    """Create a job from a JSON body (no file upload)."""
    job = Job(
        workflow=body.workflow,
        prompt=body.prompt,
        params=body.params,
    )
    session.add(job)
    await session.commit()

    await run_job.kiq(job.id)
    logger.info("Enqueued job %s (workflow=%s)", job.id[:8], body.workflow)
    return JobResponse.from_job(job)


@app.get("/jobs", response_model=list[JobResponse])
async def list_jobs(
    status: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[JobResponse]:
    """List jobs, optionally filtered by status."""
    stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(Job.status == status)
    result = await session.execute(stmt)
    return [JobResponse.from_job(j) for j in result.scalars()]


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    """Get a single job by ID."""
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse.from_job(job)


@app.delete("/jobs/{job_id}", response_model=JobResponse)
async def cancel_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    """Cancel a pending job. Running jobs are not interrupted."""
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("pending",):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job with status '{job.status}' (only pending jobs can be cancelled)",
        )
    job.mark_cancelled()
    await session.commit()
    return JobResponse.from_job(job)


@app.get("/output/{filename}")
async def get_output(filename: str) -> FileResponse:
    """Download a generated image by filename."""
    # Sanitise: no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = _OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "broker": type(broker).__name__}


# ---------------------------------------------------------------------------
# WebSocket — per-job progress stream
# ---------------------------------------------------------------------------

# Active WebSocket connections keyed by job_id
_ws_listeners: dict[str, list[asyncio.Queue]] = {}


def _register_ws(job_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _ws_listeners.setdefault(job_id, []).append(q)
    return q


def _unregister_ws(job_id: str, q: asyncio.Queue) -> None:
    listeners = _ws_listeners.get(job_id, [])
    if q in listeners:
        listeners.remove(q)
    if not listeners:
        _ws_listeners.pop(job_id, None)


async def broadcast_progress(job_id: str, event: dict) -> None:
    """Called by the task worker to push progress events to WebSocket clients."""
    for q in list(_ws_listeners.get(job_id, [])):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


@app.websocket("/ws/{job_id}")
async def ws_progress(
    websocket: WebSocket,
    job_id: str,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Stream progress events for a job in real-time."""
    await websocket.accept()

    job = await session.get(Job, job_id)
    if job is None:
        await websocket.send_json({"event": "error", "data": {"message": "Job not found"}})
        await websocket.close(code=4004)
        return

    await websocket.send_json({"event": "status", "data": {"status": job.status}})
    if job.status in ("done", "failed", "cancelled"):
        await websocket.close()
        return

    q = _register_ws(job_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Ping to keep connection alive
                await websocket.send_json({"event": "ping"})
                continue

            await websocket.send_json(event)

            # Terminal events — close after sending
            if event.get("event") in ("execution_success", "execution_error", "execution_interrupted"):
                break

    except WebSocketDisconnect:
        pass
    finally:
        _unregister_ws(job_id, q)


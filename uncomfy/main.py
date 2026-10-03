from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .auth import (
    create_access_token,
    get_current_user,
    get_optional_user,
    get_user_by_username,
    hash_password,
    verify_password,
)
from .database import get_session, init_db
from .models import Job, User
from .tasks import broker, run_job

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_TEMPLATES_DIR = _HERE / "templates"
_STATIC_DIR = _HERE / "static"
_UPLOAD_DIR = _HERE.parent / "uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)
_OUTPUT_DIR = Path(os.getenv("COMFY_PATH", r"C:\ComfyUI")) / "output"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    await init_db()
    await broker.startup()
    yield
    await broker.shutdown()


app = FastAPI(title="uncomfy", version="0.1.0", lifespan=_lifespan)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

JobStatus = Literal["pending", "running", "done", "failed", "cancelled"]


class JobCreate(BaseModel):
    workflow: str = Field(default="img2img")
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
    owner_id: str | None = None

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
            owner_id=job.owner_id,
        )


class UserResponse(BaseModel):
    id: str
    username: str
    is_active: bool
    created_at: str

    @classmethod
    def from_user(cls, u: User) -> "UserResponse":
        return cls(
            id=u.id,
            username=u.username,
            is_active=u.is_active,
            created_at=u.created_at.isoformat(),
        )


# ---------------------------------------------------------------------------
# Auth routes (JSON API)
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=UserResponse, status_code=201)
async def register(
    username: str = Form(...),
    password: str = Form(...),
    session: AsyncSession = Depends(get_session),
) -> UserResponse:
    if len(username) < 3:
        raise HTTPException(status_code=422, detail="Username must be at least 3 characters")
    if len(password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters")
    existing = await get_user_by_username(session, username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")
    user = User(username=username, hashed_password=hash_password(password))
    session.add(user)
    await session.commit()
    return UserResponse.from_user(user)


@app.post("/auth/token")
async def token(
    form: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """OAuth2 password flow — returns bearer token."""
    user = await get_user_by_username(session, form.username)
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
    token_str = create_access_token(user.id, user.username)
    return {"access_token": token_str, "token_type": "bearer"}


@app.get("/auth/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.from_user(current_user)


# ---------------------------------------------------------------------------
# HTML auth routes (cookie-based)
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    user = await get_user_by_username(session, username)
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            request, "login.html", {"error": "Неверный логин или пароль"}, status_code=401
        )
    token_str = create_access_token(user.id, user.username)
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie(
        "access_token",
        token_str,
        httponly=True,
        samesite="lax",
        max_age=3600,
    )
    return resp


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "register.html", {"error": None})


@app.post("/register", response_class=HTMLResponse)
async def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    if len(username) < 3 or len(password) < 6:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "Логин ≥ 3 символа, пароль ≥ 6 символов"},
            status_code=422,
        )
    existing = await get_user_by_username(session, username)
    if existing:
        return templates.TemplateResponse(
            request, "register.html", {"error": "Имя занято"}, status_code=409
        )
    user = User(username=username, hashed_password=hash_password(password))
    session.add(user)
    await session.commit()
    token_str = create_access_token(user.id, user.username)
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie("access_token", token_str, httponly=True, samesite="lax", max_age=3600)
    return resp


@app.post("/logout")
async def logout() -> RedirectResponse:
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("access_token")
    return resp


# ---------------------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    current_user: User | None = Depends(get_optional_user),
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)

    stmt = (
        select(Job)
        .where(Job.owner_id == current_user.id)
        .order_by(Job.created_at.desc())
        .limit(50)
    )
    result = await session.execute(stmt)
    jobs = result.scalars().all()
    return templates.TemplateResponse(
        request, "index.html", {"user": current_user, "jobs": jobs}
    )


@app.get("/jobs/{job_id}/view", response_class=HTMLResponse)
async def job_detail_page(
    request: Request,
    job_id: str,
    current_user: User | None = Depends(get_optional_user),
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    if current_user is None:
        return RedirectResponse(url="/login", status_code=303)
    job = await session.get(Job, job_id)
    if job is None or job.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    return templates.TemplateResponse(
        request, "job.html", {"user": current_user, "job": job}
    )


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

async def _save_upload(file: UploadFile) -> Path:
    ext = Path(file.filename or "upload.png").suffix or ".png"
    dest = _UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    content = await file.read()
    dest.write_bytes(content)
    return dest


# ---------------------------------------------------------------------------
# JSON API — Jobs
# ---------------------------------------------------------------------------

@app.post("/jobs", response_model=JobResponse, status_code=202)
async def create_job(
    workflow: str = Query(default="img2img"),
    prompt: str | None = Query(default=None),
    image: UploadFile | None = File(default=None),
    current_user: User | None = Depends(get_optional_user),
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    input_path: str | None = None
    if image is not None:
        saved = await _save_upload(image)
        input_path = str(saved)

    job = Job(
        workflow=workflow,
        prompt=prompt,
        input_path=input_path,
        owner_id=current_user.id if current_user else None,
    )
    session.add(job)
    await session.commit()

    await run_job.kiq(job.id)
    logger.info("Enqueued job %s (workflow=%s)", job.id[:8], workflow)
    return JobResponse.from_job(job)


@app.post("/jobs/json", response_model=JobResponse, status_code=202)
async def create_job_json(
    body: JobCreate,
    current_user: User | None = Depends(get_optional_user),
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    job = Job(
        workflow=body.workflow,
        prompt=body.prompt,
        params=body.params,
        owner_id=current_user.id if current_user else None,
    )
    session.add(job)
    await session.commit()

    await run_job.kiq(job.id)
    logger.info("Enqueued job %s (workflow=%s)", job.id[:8], body.workflow)
    return JobResponse.from_job(job)


@app.get("/jobs", response_model=list[JobResponse])
async def list_jobs(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[JobResponse]:
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
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse.from_job(job)


@app.delete("/jobs/{job_id}", response_model=JobResponse)
async def cancel_job(
    job_id: str,
    session: AsyncSession = Depends(get_session),
) -> JobResponse:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("pending",):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job with status '{job.status}'",
        )
    job.mark_cancelled()
    await session.commit()
    return JobResponse.from_job(job)


@app.get("/output/{filename}")
async def get_output(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = _OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "broker": type(broker).__name__}


# ---------------------------------------------------------------------------
# WebSocket — per-job progress stream
# ---------------------------------------------------------------------------

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
                await websocket.send_json({"event": "ping"})
                continue

            await websocket.send_json(event)

            if event.get("event") in ("execution_success", "execution_error", "execution_interrupted"):
                break

    except WebSocketDisconnect:
        pass
    finally:
        _unregister_ws(job_id, q)

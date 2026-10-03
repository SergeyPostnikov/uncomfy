"""
Integration tests for FastAPI routes — no ComfyUI, no real broker.
"""

from __future__ import annotations

import io
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from uncomfy.models import Job
from uncomfy.database import Base

# ---------------------------------------------------------------------------
# In-memory SQLite test database
# ---------------------------------------------------------------------------

TEST_DB = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="module")
async def db_engine():
    engine = create_async_engine(TEST_DB)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    Session = async_sessionmaker(db_engine, expire_on_commit=False)
    async with Session() as session:
        yield session


# ---------------------------------------------------------------------------
# App fixture — override deps + broker
# ---------------------------------------------------------------------------

@pytest.fixture
async def app(db_engine):
    from uncomfy import main as m
    from uncomfy.database import get_session

    Session = async_sessionmaker(db_engine, expire_on_commit=False)

    async def _override_session():
        async with Session() as s:
            yield s

    m.app.dependency_overrides[get_session] = _override_session

    # Patch broker startup/shutdown and run_job.kiq so no real task queue needed
    with (
        patch.object(m.broker, "startup", new=AsyncMock()),
        patch.object(m.broker, "shutdown", new=AsyncMock()),
    ):
        yield m.app

    m.app.dependency_overrides.clear()


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    async def test_health_ok(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "broker" in data


# ---------------------------------------------------------------------------
# POST /jobs/json
# ---------------------------------------------------------------------------

class TestCreateJobJson:
    async def test_creates_pending_job(self, client):
        with patch("uncomfy.main.run_job") as mock_task:
            mock_task.kiq = AsyncMock()
            r = await client.post("/jobs/json", json={"workflow": "img2img", "prompt": "test"})

        assert r.status_code == 202
        data = r.json()
        assert data["status"] == "pending"
        assert data["workflow"] == "img2img"
        assert data["prompt"] == "test"
        assert "id" in data

    async def test_enqueues_task(self, client):
        with patch("uncomfy.main.run_job") as mock_task:
            mock_task.kiq = AsyncMock(return_value=None)
            r = await client.post("/jobs/json", json={"workflow": "test_wf"})
            assert r.status_code == 202
            mock_task.kiq.assert_called_once()

    async def test_with_params(self, client):
        with patch("uncomfy.main.run_job") as mock_task:
            mock_task.kiq = AsyncMock()
            r = await client.post(
                "/jobs/json",
                json={"workflow": "img2img", "params": {"steps": 20, "denoise": 0.8}},
            )
        assert r.status_code == 202
        assert r.json()["params"]["steps"] == 20


# ---------------------------------------------------------------------------
# GET /jobs/{id}
# ---------------------------------------------------------------------------

class TestGetJob:
    async def test_returns_job(self, client, db_session):
        job = Job(id=str(uuid.uuid4()), workflow="test", status="pending")
        db_session.add(job)
        await db_session.commit()

        with patch("uncomfy.main.run_job") as mock_task:
            mock_task.kiq = AsyncMock()
            r = await client.get(f"/jobs/{job.id}")

        assert r.status_code == 200
        assert r.json()["id"] == job.id

    async def test_404_for_unknown(self, client):
        r = await client.get(f"/jobs/{uuid.uuid4()}")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /jobs
# ---------------------------------------------------------------------------

class TestListJobs:
    async def test_returns_list(self, client, db_session):
        for _ in range(3):
            db_session.add(Job(id=str(uuid.uuid4()), workflow="wf", status="done"))
        await db_session.commit()

        r = await client.get("/jobs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    async def test_filter_by_status(self, client, db_session):
        job = Job(id=str(uuid.uuid4()), workflow="wf", status="failed")
        db_session.add(job)
        await db_session.commit()

        r = await client.get("/jobs?status=failed")
        assert r.status_code == 200
        statuses = [j["status"] for j in r.json()]
        assert all(s == "failed" for s in statuses)

    async def test_limit_respected(self, client, db_session):
        for _ in range(5):
            db_session.add(Job(id=str(uuid.uuid4()), workflow="wf", status="pending"))
        await db_session.commit()

        r = await client.get("/jobs?limit=2")
        assert r.status_code == 200
        assert len(r.json()) <= 2


# ---------------------------------------------------------------------------
# DELETE /jobs/{id}
# ---------------------------------------------------------------------------

class TestCancelJob:
    async def test_cancel_pending(self, client, db_session):
        job = Job(id=str(uuid.uuid4()), workflow="wf", status="pending")
        db_session.add(job)
        await db_session.commit()

        r = await client.delete(f"/jobs/{job.id}")
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

    async def test_cannot_cancel_running(self, client, db_session):
        job = Job(id=str(uuid.uuid4()), workflow="wf", status="running")
        db_session.add(job)
        await db_session.commit()

        r = await client.delete(f"/jobs/{job.id}")
        assert r.status_code == 409

    async def test_cannot_cancel_done(self, client, db_session):
        job = Job(id=str(uuid.uuid4()), workflow="wf", status="done")
        db_session.add(job)
        await db_session.commit()

        r = await client.delete(f"/jobs/{job.id}")
        assert r.status_code == 409

    async def test_404_for_unknown(self, client):
        r = await client.delete(f"/jobs/{uuid.uuid4()}")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /output/{filename}
# ---------------------------------------------------------------------------

class TestGetOutput:
    async def test_rejects_path_traversal(self, client):
        r = await client.get("/output/../secret.txt")
        assert r.status_code in (400, 404)

    async def test_rejects_backslash(self, client):
        r = await client.get("/output/foo\\bar.png")
        assert r.status_code == 400

    async def test_404_for_missing_file(self, client):
        r = await client.get("/output/nonexistent_999.png")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# WebSocket /ws/{job_id}
# ---------------------------------------------------------------------------

class TestWebSocket:
    async def test_404_for_unknown_job(self, app):
        from fastapi.testclient import TestClient
        with TestClient(app) as tc:
            with tc.websocket_connect(f"/ws/{uuid.uuid4()}") as ws:
                msg = ws.receive_json()
                assert msg["event"] == "error"

    async def test_immediate_close_for_done_job(self, app, db_session):
        job = Job(id=str(uuid.uuid4()), workflow="wf", status="done")
        db_session.add(job)
        await db_session.commit()

        from fastapi.testclient import TestClient
        with TestClient(app) as tc:
            with tc.websocket_connect(f"/ws/{job.id}") as ws:
                msg = ws.receive_json()
                assert msg["event"] == "status"
                assert msg["data"]["status"] == "done"

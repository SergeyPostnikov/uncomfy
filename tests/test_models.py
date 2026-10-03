"""
Tests for database models and session — uses in-memory SQLite, no GPU needed.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from uncomfy.database import Base
from uncomfy.models import Job


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


# ---------------------------------------------------------------------------
# Job creation
# ---------------------------------------------------------------------------

class TestJobCreation:
    async def test_default_status_is_pending(self, session):
        job = Job(workflow="img2img")
        session.add(job)
        await session.commit()
        await session.refresh(job)
        assert job.status == "pending"

    async def test_id_is_uuid_string(self, session):
        job = Job(workflow="img2img")
        session.add(job)
        await session.commit()
        assert len(job.id) == 36
        assert job.id.count("-") == 4

    async def test_created_at_set_automatically(self, session):
        job = Job(workflow="t2i")
        session.add(job)
        await session.commit()
        assert isinstance(job.created_at, datetime)

    async def test_optional_fields_nullable(self, session):
        job = Job(workflow="img2img")
        session.add(job)
        await session.commit()
        assert job.prompt is None
        assert job.params is None
        assert job.input_path is None
        assert job.output_path is None
        assert job.error is None
        assert job.duration_sec is None

    async def test_params_stored_as_json(self, session):
        params = {"steps": 20, "denoise": 0.85, "seed": 42}
        job = Job(workflow="img2img", params=params)
        session.add(job)
        await session.commit()
        await session.refresh(job)
        assert job.params["steps"] == 20
        assert job.params["denoise"] == 0.85

    async def test_repr(self, session):
        job = Job(workflow="img2img")
        session.add(job)
        await session.commit()
        r = repr(job)
        assert "pending" in r
        assert "img2img" in r


# ---------------------------------------------------------------------------
# Lifecycle transitions
# ---------------------------------------------------------------------------

class TestJobLifecycle:
    async def test_mark_running(self, session):
        job = Job(workflow="img2img")
        session.add(job)
        await session.commit()

        job.mark_running()
        await session.commit()

        assert job.status == "running"
        assert job.started_at is not None

    async def test_mark_done(self, session):
        job = Job(workflow="img2img")
        session.add(job)
        await session.commit()

        job.mark_running()
        job.mark_done("/output/result.png")
        await session.commit()

        assert job.status == "done"
        assert job.output_path == "/output/result.png"
        assert job.finished_at is not None
        assert job.duration_sec is not None
        assert job.duration_sec >= 0

    async def test_mark_failed(self, session):
        job = Job(workflow="img2img")
        session.add(job)
        await session.commit()

        job.mark_running()
        job.mark_failed("CUDA out of memory")
        await session.commit()

        assert job.status == "failed"
        assert job.error == "CUDA out of memory"
        assert job.finished_at is not None
        assert job.duration_sec is not None

    async def test_mark_cancelled(self, session):
        job = Job(workflow="img2img")
        session.add(job)
        await session.commit()

        job.mark_cancelled()
        await session.commit()

        assert job.status == "cancelled"
        assert job.finished_at is not None

    async def test_duration_none_if_never_started(self, session):
        job = Job(workflow="img2img")
        session.add(job)
        await session.commit()

        job.mark_failed("error before start")
        assert job.duration_sec is None

    async def test_done_without_started_at(self, session):
        job = Job(workflow="img2img")
        session.add(job)
        await session.commit()

        job.mark_done("/output/x.png")
        assert job.duration_sec is None


# ---------------------------------------------------------------------------
# Persistence queries
# ---------------------------------------------------------------------------

class TestJobQueries:
    async def test_filter_by_status(self, session):
        for status, wf in [("pending", "t2i"), ("running", "img2img"), ("done", "t2i")]:
            job = Job(workflow=wf)
            job.status = status
            session.add(job)
        await session.commit()

        from sqlalchemy import select
        result = await session.execute(select(Job).where(Job.status == "pending"))
        pending = result.scalars().all()
        assert len(pending) == 1
        assert pending[0].workflow == "t2i"

    async def test_multiple_jobs_persist(self, session):
        for i in range(5):
            session.add(Job(workflow="img2img", prompt=f"prompt {i}"))
        await session.commit()

        from sqlalchemy import select, func
        result = await session.execute(select(func.count()).select_from(Job))
        assert result.scalar() == 5

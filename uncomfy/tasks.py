from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from taskiq import AsyncBroker, InMemoryBroker, TaskiqMiddleware
from taskiq_redis import ListQueueBroker

from .database import SessionLocal
from .models import Job

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Broker — Redis in production, InMemory for local/tests
# ---------------------------------------------------------------------------

def _make_broker() -> AsyncBroker:
    url = os.getenv("REDIS_URL")
    if url:
        return ListQueueBroker(url)
    logger.warning("REDIS_URL not set — using InMemoryBroker (no persistence)")
    return InMemoryBroker()


broker = _make_broker()

# ---------------------------------------------------------------------------
# Engine singleton — initialised once per worker process
# ---------------------------------------------------------------------------

_engine = None


async def _get_engine():
    global _engine
    if _engine is None:
        from .engine import InferenceEngine
        _engine = InferenceEngine()
        await _engine.initialize()
    return _engine


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@broker.task
async def run_job(job_id: str) -> dict:
    """
    Execute a ComfyUI workflow for the given job_id.

    Reads job from DB, runs inference, writes result back.
    Returns a minimal status dict.
    """
    async with SessionLocal() as session:
        job = await session.get(Job, job_id)
        if job is None:
            logger.error("Job %s not found in DB", job_id)
            return {"status": "not_found", "job_id": job_id}

        if job.status == "cancelled":
            return {"status": "cancelled", "job_id": job_id}

        job.mark_running()
        await session.commit()

    try:
        engine = await _get_engine()

        workflow = _load_workflow(job)
        result = await engine.run(workflow, prompt_id=job_id)
        paths = engine.output_images(result)
        output_path = str(paths[0]) if paths else ""

        async with SessionLocal() as session:
            job = await session.get(Job, job_id)
            job.mark_done(output_path)
            await session.commit()

        logger.info("Job %s done in %.1fs → %s", job_id[:8], job.duration_sec, output_path)
        return {"status": "done", "job_id": job_id, "output_path": output_path}

    except Exception as exc:
        error_msg = str(exc)
        logger.exception("Job %s failed: %s", job_id[:8], error_msg)

        async with SessionLocal() as session:
            job = await session.get(Job, job_id)
            if job:
                job.mark_failed(error_msg)
                await session.commit()

        return {"status": "failed", "job_id": job_id, "error": error_msg}


def _load_workflow(job: Job) -> dict:
    """
    Build the workflow dict from job fields.

    Priority:
    1. job.params["workflow_json"] — inline dict (API clients)
    2. job.params["workflow_file"] — path to JSON file in workflows/
    3. job.workflow              — name matching workflows/<name>.json
    """
    params = job.params or {}

    if "workflow_json" in params:
        return params["workflow_json"]

    workflows_dir = Path(__file__).parent.parent / "uncomfy" / "workflows"

    filename = params.get("workflow_file") or f"{job.workflow}.json"
    path = Path(filename) if Path(filename).is_absolute() else workflows_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {path}")

    with path.open(encoding="utf-8") as f:
        workflow = json.load(f)

    # Patch prompt and input image if provided
    _patch_workflow(workflow, job)
    return workflow


def _patch_workflow(workflow: dict, job: Job) -> None:
    """Apply job.prompt and job.input_path into the workflow nodes."""
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if ct == "TextEncodeQwenImage21" and job.prompt:
            inputs["prompt"] = job.prompt

        if ct == "LoadImage" and job.input_path:
            inputs["image"] = job.input_path

        if ct == "KSampler" and job.params:
            for key in ("steps", "denoise", "cfg", "seed", "sampler_name", "scheduler"):
                if key in (job.params or {}):
                    inputs[key] = job.params[key]

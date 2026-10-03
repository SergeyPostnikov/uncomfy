"""
Thin wrapper around ComfyUI's PromptExecutor.

Implements the ExecutionServer protocol so the executor has something
to talk to — events are collected in memory instead of broadcast over
aiohttp websockets.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

COMFY_PATH = Path(r"C:\ComfyUI")


def _bootstrap(comfy_path: Path = COMFY_PATH) -> None:
    """
    Must be called before any comfy import.
    Inserts ComfyUI into sys.path and initialises CLI arg defaults
    so that comfy.cli_args doesn't try to parse our argv.
    """
    root = str(comfy_path)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Windows single-GPU guard (mirrors ComfyUI main.py)
    if os.name == "nt" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import comfy.options
    comfy.options.enable_args_parsing()


# ---------------------------------------------------------------------------
# ExecutionServer stub
# ---------------------------------------------------------------------------

class StubServer:
    """
    Minimal ExecutionServer protocol implementation.

    PromptExecutor calls send_sync() for every lifecycle event
    (execution_start, progress, execution_success, execution_error …).
    We collect those in self.events and optionally push them to an
    asyncio.Queue for streaming to callers.
    """

    client_id: str | None = None
    last_node_id: str | None = None
    sockets_metadata: dict[str, dict[str, Any]] = {}

    # Events that are interesting enough to push to a progress queue.
    _PROGRESS_EVENTS = frozenset({
        "progress",
        "executing",
        "execution_start",
        "execution_cached",
        "execution_success",
        "execution_error",
        "execution_interrupted",
    })

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._progress_queue: asyncio.Queue[dict] | None = None

    # -- ExecutionServer protocol --

    def send_sync(
        self,
        event: str | int,
        data: object,
        sid: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {"event": event, "data": data}
        self.events.append(entry)
        if self._progress_queue is not None and event in self._PROGRESS_EVENTS:
            try:
                self._progress_queue.put_nowait(entry)
            except asyncio.QueueFull:
                logger.debug("progress queue full, dropping event %s", event)

    def queue_updated(self) -> None:
        pass

    # -- Helpers --

    def attach_progress_queue(self, q: asyncio.Queue[dict]) -> None:
        self._progress_queue = q

    def detach_progress_queue(self) -> None:
        self._progress_queue = None

    def clear(self) -> None:
        self.events.clear()

    def last_error(self) -> dict | None:
        for e in reversed(self.events):
            if e["event"] == "execution_error":
                return e["data"]
        return None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Async wrapper around ComfyUI's PromptExecutor.

    Usage::

        engine = InferenceEngine()
        await engine.initialize()

        result = await engine.run(workflow_dict)
        paths  = engine.output_images(result)
    """

    def __init__(self, comfy_path: Path = COMFY_PATH) -> None:
        self.comfy_path = comfy_path
        self.server = StubServer()
        self._executor: Any = None  # PromptExecutor — imported lazily
        self._ready = False

    # -- Lifecycle --

    async def initialize(
        self,
        *,
        load_custom_nodes: bool = True,
        load_api_nodes: bool = True,
    ) -> None:
        """
        Load ComfyUI nodes and create the PromptExecutor.
        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._ready:
            return

        _bootstrap(self.comfy_path)

        import nodes as comfy_nodes
        import execution
        import comfy.model_management
        import comfy.utils

        await comfy_nodes.init_extra_nodes(
            init_custom_nodes=load_custom_nodes,
            init_api_nodes=load_api_nodes,
        )

        comfy.model_management.set_cudnn_benchmark()
        self._install_progress_hook(comfy.model_management, comfy.utils)

        ram_gb = comfy.model_management.total_ram / 1024.0
        cache_ram = min(10.0, max(2.0, ram_gb * 0.10))
        cache_ram_inactive = min(128.0, ram_gb)

        self._executor = execution.PromptExecutor(
            self.server,
            cache_type=execution.CacheType.RAM_PRESSURE,
            cache_args={
                "lru": 0,
                "ram": cache_ram,
                "ram_inactive": cache_ram_inactive,
            },
        )

        self._ready = True
        logger.info("InferenceEngine ready (cache ram=%.1fGB inactive=%.1fGB)", cache_ram, cache_ram_inactive)

    def _install_progress_hook(self, model_management: Any, comfy_utils: Any) -> None:
        """Replace ComfyUI's tqdm progress hook with our send_sync bridge."""
        server = self.server

        def _hook(value, total, preview_image, prompt_id=None, node_id=None):
            model_management.throw_exception_if_processing_interrupted()
            server.send_sync(
                "progress",
                {"value": value, "max": total, "prompt_id": prompt_id, "node": node_id},
            )

        comfy_utils.set_progress_bar_global_hook(_hook)

    # -- Execution --

    async def run(
        self,
        workflow: dict,
        *,
        prompt_id: str | None = None,
        extra_data: dict | None = None,
        progress_queue: asyncio.Queue[dict] | None = None,
    ) -> dict:
        """
        Execute a workflow dict (ComfyUI API format).

        Returns history_result dict with ``outputs`` key on success.
        Raises RuntimeError on execution failure.
        """
        if not self._ready or self._executor is None:
            raise RuntimeError("Engine not initialised — call await engine.initialize() first")

        prompt_id = prompt_id or str(uuid.uuid4())
        self.server.clear()

        if progress_queue is not None:
            self.server.attach_progress_queue(progress_queue)

        try:
            await self._executor.execute_async(
                workflow,
                prompt_id,
                extra_data=extra_data or {},
            )
        finally:
            self.server.detach_progress_queue()

        if not self._executor.success:
            err = self.server.last_error()
            msg = err.get("exception_message", "unknown error") if err else "execution failed"
            raise RuntimeError(f"Execution failed [{prompt_id}]: {msg}")

        return self._executor.history_result

    # -- Utilities --

    def output_images(self, history_result: dict) -> list[Path]:
        """Return absolute paths for all images in a history_result."""
        import folder_paths
        out_dir = Path(folder_paths.get_output_directory())
        paths: list[Path] = []
        for node_output in history_result.get("outputs", {}).values():
            for img in node_output.get("images", []):
                subfolder = img.get("subfolder", "")
                fname = img["filename"]
                paths.append(out_dir / subfolder / fname if subfolder else out_dir / fname)
        return paths

    @property
    def ready(self) -> bool:
        return self._ready
